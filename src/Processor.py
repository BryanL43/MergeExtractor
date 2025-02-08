import re
from concurrent.futures import ThreadPoolExecutor, as_completed, Future, wait, ALL_COMPLETED
from threading import Event, Lock
import sys
import requests
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import os
import shutil
import random
import time

from Assistant import Assistant
from Logger import Logger
from Document import Document

TEMP_DIRECTORY = "temp";

"""
    - This object handles processing the document by:
        - Cleaning the data
        - Verifying the existence of both companies are present
        - Extracting the background section
"""
class Processor:
    def __init__(self, assistant: Assistant, nlp: any, threadCount: int, startPhrases: list, executor: ThreadPoolExecutor):
        self.assistant = assistant;
        self.nlp = nlp;
        self.threadCount = threadCount;
        self.startPhrases = startPhrases;

        self.executor = executor;
        self.__terminationEvent = Event(); # Stops the multithreading
    
        print("Successfully initialized Processor");

    def __extractFirstWord(self, companyName) -> str:
        clean_name = re.sub(r"\(.*?\)", "", companyName);  # Remove parentheses content
        first_word = re.split(r"[\s\-_]", clean_name.strip())[0];
        return first_word;

    def __loadFileFromURL(self, url) -> str:
        # Create a request that mimics browser activity
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }

        response = requests.get(url, headers=headers);
        if (response.text):
            return response.text;
        else:
            print(f"FATAL: Failed to load document via url. Err_Code: {response.status_code}");
            sys.exit(response.status_code);

    def __preProcessText(self, content) -> str:
        soup = BeautifulSoup(content, "html.parser");
        text = soup.get_text(separator="\n");

        # Remove standalone page numbers
        pageNumPattern = re.compile(r'^\s*\d+\s*$', re.MULTILINE);
        text = re.sub(pageNumPattern, '', text);

        # Remove extra newline characters
        text = re.sub(r'\n\s*\n+', '\n\n', text);

        return text.strip();

    def __checkCompaniesInDocument(self, url, companyNames) -> tuple[str, bool]:
        # Open the url and acquire the document content.
        # Error = Fatal, force exit from load function.
        rawText = self.__loadFileFromURL(url);

        cleanedText = self.__preProcessText(rawText);
        lowerText = cleanedText.lower();

        # Check if both company names are present as whole words
        foundCompanies = [name for name in companyNames if re.search(r'\b' + re.escape(name) + r'\b', lowerText)];
        
        # Return the cleanedText if both company names are found, else False
        return cleanedText, len(foundCompanies) == len(companyNames);

    def __removeTableOfContents(self, text) -> str:
        # Regular expression patterns for table of contents
        tocStartPattern = re.compile(r'(Table of Contents|Contents|TABLE OF CONTENT|CONTENTS)', re.IGNORECASE);
        tocEndPattern = re.compile(r'(Introduction|Chapter \d+|Section \d+|Part \d+|Page \d+)', re.IGNORECASE);

        # Find the start of the table of contents
        tocStartMatch = tocStartPattern.search(text);
        if (not tocStartMatch):  # No table of contents found
            return text;

        tocStartIndex = tocStartMatch.start();

        # Find the end of the table of contents
        tocEndMatch = tocEndPattern.search(text, tocStartIndex);
        if (not tocEndMatch):  # No end of table of contents found
            return text;

        tocEndIndex = tocEndMatch.start();

        # Remove the table of contents section
        cleanedText = text[:tocStartIndex] + text[tocEndIndex:];

        # Remove any remaining table of contents references
        cleanedText = re.sub(r'\btable\s*of\s*contents?\b|\btableofcontents?\b', '', cleanedText, flags=re.IGNORECASE);
        cleanedText = re.sub(r'(?i)table\s*of\s*contents?|tableofcontents?', '', cleanedText);

        return cleanedText.strip();

    def getDocuments(self, sourceLinks: list, companyNames: list) -> list[Document]:
        # Acquire company name's first word
        companyNamesCut = [self.__extractFirstWord(name).lower() for name in companyNames];

        # Create multiple threads to open & verify document
        futures = {self.executor.submit(self.__checkCompaniesInDocument, url, companyNamesCut): url for url in sourceLinks};

        # Race conditions causing no results for ones that should have results
        wait([future for future in futures], return_when=ALL_COMPLETED);

        # Wait for thread to finish processing and create new Document object
        documents = [];
        for future in as_completed(futures):
            url = futures[future];
            try:
                cleanedText, bothFound = future.result();
                if bothFound:
                    documents.append(Document(url, cleanedText));
            except Exception as e:
                Logger.logMessage(f"[{Logger.get_current_timestamp()}] [-] Error retrieving document for URL {url}: {e}");
    
        return documents;

    def __findSection(self, doc: Document, startCandidates: list[str]) -> bool:
        # Clean and truncate text for spaCy
        cleanedText = self.__removeTableOfContents(doc.getContent());
        doc.setContent(cleanedText[1000:1000000]); # Preserve some content for assistant parsing
        cleanedText = cleanedText[50000:1000000];  # Shrink to manageable size for spaCy

        # Break the text into sentences
        doc = self.nlp(cleanedText);
        sentences = [sent.text for sent in doc.sents];

        # Locate the start of the desired background section
        for i, sentence in enumerate(sentences):
            match = next(
                (sc for sc in startCandidates if sc.lower() in sentence.lower() or fuzz.partial_ratio(sentence.lower(), sc.lower()) > 90),
                None
            );
            if match:
                return True;
        
        return False;

    # Helper function for fallback method
    def __analyzeDocumentWithObj(self, file_path: str, doc: Document):
        result = self.assistant.analyzeDocument(file_path);
        return result, doc;

    def __processFallbackFutures(self, futures: list[Future]) -> (Document | None):
        # Wait for all futures to complete
        wait(futures, return_when=ALL_COMPLETED);
    
        sectionFoundEvent = Event(); # Localized as we need to wait for all futures to complete

        for future in as_completed(futures):
            if sectionFoundEvent.is_set():
                break;

            try:
                result, doc = future.result();
                if result is None:
                    continue;

                match = re.search(r"\[(.*?)\]", result);
                foundSection = match.group(1) if match else "unknown";

                if foundSection == "Found":
                    sectionFoundEvent.set();  # Signal that a section has been found
                    return doc; # Returns the correct document with the detected section
            except Exception as e:
                Logger.logMessage(f"[{Logger.get_current_timestamp()}] [-] Error processing fallback future: {e}");
    
        return None;

    def locateDocument(self, documents: list[Document], companyNames: list, mainIndex: int) -> (str | None):
        """
            - Acquires the document with the "Background of the Merger" section and return its url.
        """

        formatDocName = f"{mainIndex}_{companyNames[0].replace(' ', '_')}_&_{companyNames[1].replace(' ', '_')}";
        
        # Multi-thread break variables
        foundData = False;
        self.__terminationEvent.clear();
        lock = Lock();  # Ensure only one thread writes

        # Concurrently process all documents & locate "Background of the Merger" section via fuzzy matching
        futures = {self.executor.submit(self.__findSection, doc, self.startPhrases): doc for doc in documents};
        
        for future in as_completed(futures):
            if self.__terminationEvent.is_set():  # If background section is found already
                break;

            try:
                doc = futures[future];
                if future.result():  # If "Background of the Merger" section is found
                    if not foundData: # First check before locking (avoids unnecessary contention)
                        with lock:
                            if not foundData: # Back up check for optimization
                                self.__terminationEvent.set();
                                foundData = True;

                                # Write the data to a file
                                batchStart = (mainIndex // 100) * 100;
                                batchEnd = batchStart + 99;
                                with open(f"./DataSet/{batchStart}-{batchEnd}/{formatDocName}.txt", "w", encoding="utf-8") as file:
                                    file.write(f"URL: {doc.getUrl()}\n\n");
                                    file.write(doc.getContent());

                                Logger.logMessage(f"[{Logger.get_current_timestamp()}] [+] Successfully created document for: {companyNames[0]} & {companyNames[1]}");
                                return doc.getUrl();

                    break;
            except Exception as e:
                url = futures[future];
                Logger.logMessage(f"[{Logger.get_current_timestamp()}] [-] Error processing {url}: {e}");

        # Fallback method if fuzzy fails. We will use openai to determine if the background section is within any of the documents
        if not foundData:
            Logger.logMessage(f"[{Logger.get_current_timestamp()}] [*] No background section found for index {mainIndex}: {companyNames[0]} & {companyNames[1]}. Retrying via fallback...");

            # Create temp directory for creating temp file acceptable by openai
            os.makedirs(TEMP_DIRECTORY, exist_ok=True);
            fallbackFutures = [];

            # Create a list of async processes to force correct using openai
            for doc in documents:
                filePath = os.path.join(TEMP_DIRECTORY, f"temp_{random.randint(1000, 99999)}.txt");
                with open(filePath, "w", encoding="utf-8") as file:
                    file.write(doc.getContent());
                    file.flush();
                
                # Wait for the file to be created; prevent race condition
                while not os.path.exists(filePath):
                    time.sleep(0.1);
                
                fallbackFutures.append(self.executor.submit(self.__analyzeDocumentWithObj, filePath, doc));
            
            fallbackResult = self.__processFallbackFutures(fallbackFutures);
            if fallbackResult is not None:
                # Write the data to a file
                batchStart = (mainIndex // 100) * 100;
                batchEnd = batchStart + 99;
                with open(f"./DataSet/{batchStart}-{batchEnd}/{formatDocName}.txt", "w", encoding="utf-8") as file:
                    file.write(f"URL: {fallbackResult.getUrl()}\n\n");
                    file.write(fallbackResult.getContent());
                
                if os.path.exists(TEMP_DIRECTORY):
                    shutil.rmtree(TEMP_DIRECTORY);

                Logger.logMessage(f"[{Logger.get_current_timestamp()}] [+] Successfully created document for: {companyNames[0]} & {companyNames[1]}");
                return fallbackResult.getUrl();
            
            if os.path.exists(TEMP_DIRECTORY):
                shutil.rmtree(TEMP_DIRECTORY);
            return None;

        return None; # Extreme backup