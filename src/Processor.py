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
import unicodedata
from lxml import etree

from Assistant import Assistant
from Logger import Logger
from Document import Document

TEMP_DIRECTORY = "merge_extractor_temp";

"""
    - This object handles processing the document by:
        - Cleaning the data
        - Verifying the existence of both companies are present
        - Extracting the background section
"""
class Processor:
    def __init__(self, assistant: Assistant, nlp: any, startPhrases: list, executor: ThreadPoolExecutor):
        self.assistant = assistant;
        self.nlp = nlp;
        self.startPhrases = startPhrases;

        self.executor = executor;
        self.__terminationEvent = Event(); # Stops the multithreading
    
        print("Successfully initialized Processor");

    def __extractAllButLastWord(self, companyName) -> str:
        cleanName = re.sub(r"\(.*?\)", "", companyName);  # Remove parentheses content
        words = re.split(r"[\s\-_]+", cleanName.strip());  # Split by space, hyphen, or underscore

        # Domain-like terms to merge
        mergeWords = {"net", "com", "org", "co"};

        # Merge domain-like words
        for i in range(len(words) - 1):
            if words[i].lower() in mergeWords:
                words[i] = words[i] + "." + words[i + 1];
                words.pop(i + 1);
                break;

        if len(words) > 1:
            if words[-2] == "&":
                words = words[:-2]; # Remove both "&" and the last word
            else:
                words = words[:-1];  # Remove only the last word

        return " ".join(words);

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
        # Ensure content is in UTF-8
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='ignore');
        
        # HTMLParser to handle bad HTML
        parser = etree.HTMLParser(recover=True, encoding='utf-8');
        try:
            tree = etree.fromstring(content.encode('utf-8'), parser);
            text = '\n'.join(tree.xpath('.//text()'));  # Extract all text nodes
        except Exception as e:
            raise RuntimeError(f"HTML parsing error: {e}") from e;

        # Remove standalone page numbers
        pageNumPattern = re.compile(r'^\s*\d+\s*$', re.MULTILINE);
        text = re.sub(pageNumPattern, '', text);

        # Remove extra newline characters
        text = re.sub(r'\n\s*\n+', '\n\n', text);

        return text.strip();

    def __normalizeText(self, text) -> str:
        # Remove table of contents references and normalize the text
        text = unicodedata.normalize("NFKC", text);  # Normalize Unicode
        text = text.encode("ascii", "ignore").decode("ascii");
        cleanedText = re.sub(r'\btable\s*of\s*contents?\b|\btableofcontents?\b', '', text, flags=re.IGNORECASE);
        cleanedText = re.sub(r'(?i)table\s*of\s*contents?|tableofcontents?', '', cleanedText);

        return cleanedText.strip();

    def __checkCompaniesInDocument(self, url, companyNames) -> tuple[str, bool]:
        # Open the url and acquire the document content.
        # Error = Fatal, force exit from load function.
        rawText = self.__loadFileFromURL(url);

        # Clean and truncate text
        cleanedText = self.__preProcessText(rawText);
        cleanedText = self.__normalizeText(cleanedText);
        cleanedText = cleanedText[1000:450000]; # Reduce data load

        # Truncate to header to validate that we have the correct document
        lowerText = cleanedText.lower()[:10000];

        # Check if both company names are present as whole words
        foundCompanies = [name for name in companyNames if re.search(r'\b' + re.escape(name) + r'\b', lowerText)];
        
        # Return the cleanedText if both company names are found, else False
        return cleanedText, len(foundCompanies) == len(companyNames);

    def getDocuments(self, sourceLinks: list, companyNames: list) -> list[Document]:
        # Acquire company name's first word
        companyNamesCut = [self.__extractAllButLastWord(name).lower() for name in companyNames];

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
        cleanedText = doc.getContent();
        cleanedText = cleanedText[50000:450000];  # Cut some unnecessary data for spaCy

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

    def __processFallbackFutures(self, futures: list[tuple[str, Document]]) -> (Document | None):
        sectionFoundEvent = Event(); # Localized event to ttrack section discovery

        for result, doc in futures:
            if sectionFoundEvent.is_set():
                break;
        
            try:
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
                                
                                # Cancel remaining futures
                                for f in futures:
                                    f.cancel();
                                
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
            fileDocComposite = [];

            # Create a list of async processes to force correct using openai
            for doc in documents:
                filePath = os.path.join(TEMP_DIRECTORY, f"merge_extractor_temp_{random.randint(1000, 99999)}.txt");
                with open(filePath, "w", encoding="utf-8") as file:
                    file.write(doc.getContent());
                    file.flush();
                
                # Wait for the file to be created; prevent race condition
                while not os.path.exists(filePath):
                    time.sleep(0.1);
                
                fileDocComposite.append((filePath, doc));
            
            # Parallelly process all documents to locate "Background of the Merger" section via openai
            fallbackFutures = list(self.executor.map(lambda args: self.__analyzeDocumentWithObj(*args), fileDocComposite));

            fallbackResult = self.__processFallbackFutures(fallbackFutures);
            if fallbackResult is not None:
                # Write the data to a file
                batchStart = (mainIndex // 100) * 100;
                batchEnd = batchStart + 99;
                with open(f"./DataSet/{batchStart}-{batchEnd}/{formatDocName}.txt", "w", encoding="utf-8") as file:
                    file.write(f"URL: {fallbackResult.getUrl()}\n\n");
                    file.write(fallbackResult.getContent());
                
                if os.path.exists(TEMP_DIRECTORY):
                    time.sleep(0.5);
                    shutil.rmtree(TEMP_DIRECTORY);

                Logger.logMessage(f"[{Logger.get_current_timestamp()}] [+] Successfully created document for: {companyNames[0]} & {companyNames[1]}");
                return fallbackResult.getUrl();
            
            if os.path.exists(TEMP_DIRECTORY):
                time.sleep(0.5);
                shutil.rmtree(TEMP_DIRECTORY);
            return None;

        return None; # Extreme backup