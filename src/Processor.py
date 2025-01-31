import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event
import sys
import requests
from bs4 import BeautifulSoup

from Assistant import Assistant
from Logger import Logger

"""
    - This object handles processing the document by:
        - Cleaning the data
        - Verifying the existence of both companies are present
        - Extracting the background section
"""
class Processor:
    def __init__(self, assistant: Assistant, threadCount: int, mainIndex: int):
        self.assistant = assistant;
        self.threadCount = threadCount;
        self.mainIndex = mainIndex;

        self.__terminationEvent = Event(); # Stops the multithreading
    
        print("Successfully initialized Processor");

    def __extractFirstWord(self, companyName):
        clean_name = re.sub(r"\(.*?\)", "", companyName);  # Remove parentheses content
        return clean_name.split()[0];

    def __loadFileFromURL(url):
        # Create a request that mimics browser activity
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }

        response = requests.get(url, headers=headers);
        if (response.status_code == 200):
            return response.text;
        else:
            print("FATAL: Failed to load document via url.");
            sys.exit(response.status_code);

    def __preProcessText(content):
        soup = BeautifulSoup(content, "html.parser");
        text = soup.get_text(separator="\n");

        # Remove standalone page numbers
        pageNumPattern = re.compile(r'^\s*\d+\s*$', re.MULTILINE);
        text = re.sub(pageNumPattern, '', text);

        # Remove extra newline characters
        text = re.sub(r'\n\s*\n+', '\n\n', text);

        return text.strip();

    def __checkCompaniesInDocument(self, url, companyNames):
        if (self.__terminationEvent.is_set()):
            return None, False  # Exit early if thread termination is triggered
        
        rawText = self.__loadFileFromURL(url);
        if (not rawText):  # If we cannot load the document
            return "", False;

        cleanedText = self.__preProcessText(rawText);
        lowerText = cleanedText.lower();

        # Check if both company names are present as whole words
        foundCompanies = [name for name in companyNames if re.search(r'\b' + re.escape(name) + r'\b', lowerText)];
        
        # Return the cleanedText if both company names are found, else False
        return cleanedText, len(foundCompanies) == len(companyNames);

    def __removeTableOfContents(self, text):
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

    def extractSection(self, sourceLinks: list, companyNames: list):
        """
            - Here, we will verify that both company names are present in the document.
                - Reduces the amount of documents needed to be processed with NLP.
            - Next, if both company names are present, we will try and locate the "Background of the Merger"
            chronological timeline.
        """

        companyNamesCut = [self.__extractFirstWord(name).lower() for name in companyNames];
        
        # Locate the documents with both company names present.
        foundData = False;
        self.__terminationEvent.clear();
        with ThreadPoolExecutor(max_workers=self.threadCount) as executor:
            futures = {executor.submit(self.__checkCompaniesInDocument, url, companyNamesCut): url for url in sourceLinks};

            for future in as_completed(futures):
                if (self.__terminationEvent.is_set()):  # If background section is found already
                    break;

                try:
                    cleanedText, bothFound = future.result();
                    if bothFound:
                        # Additional preprocess cleaning
                        cleanedText = self.__removeTableOfContents(cleanedText);
                        truncatedText = cleanedText[50000:1000000];  # Shrink to manageable size for spaCy

                        backgroundSection = extractSection(truncatedText, startPhrases, stopPhrases);
                        if backgroundSection is None:
                            continue;

                        # Write the data to a file
                        foundData = True;
                        formatDocName = f"{companyNames[0].replace(' ', '_')}_&_{companyNames[1].replace(' ', '_')}";
                        with open(f"../DataSet/{formatDocName}.txt", "w", encoding="utf-8") as file:
                            file.write(f"URL: {futures[future]}\n\n");
                            file.write(backgroundSection);
                        
                        Logger.logMessage(f"[{Logger.get_current_timestamp()}] [+] Successfully created document for: {companyNames[0]} & {companyNames[1]}");

                        # Signal termination and exit
                        self.__terminationEvent.set();
                        break;
                except Exception as e:
                    url = futures[future];
                    Logger.logMessage(f"[{Logger.get_current_timestamp()}] [-] Error processing {url}: {e}");

        if not foundData:
            Logger.logMessage(f"[{Logger.get_current_timestamp()}] [-] No background section found for index {self.mainIndex}: {companyNames[0]} & {companyNames[1]};");
            Logger.logMessage(f"\tDumping its document links:");
            for url in sourceLinks:
                Logger.logMessage(f"\t\t{url}");