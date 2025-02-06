from datetime import datetime
import re
import requests
import sys
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from fuzzywuzzy import fuzz
import os
import csv
from tqdm import tqdm

from Logger import Logger
from Assistant import Assistant
from Processor import Processor
from Document import Document

class Crawler:
    def __init__(
            self,
            filedDate: list,
            companyAList: list,
            companyBList: list,
            startPhrases: list,
            threadCount: int,
            nlp: any,
            assistant: Assistant
        ):
        
        self.filedDate = filedDate;
        self.companyAList = companyAList;
        self.companyBList = companyBList;
        self.startPhrases = startPhrases;
        self.threadCount = threadCount;
        self.nlp = nlp;
        self.assistant = assistant;

        self.__formTypes = ["PREM14A", "S-4", "SC 14D9", "SC TO-T"];

        print("Successfully initialized Crawler");

    # Acquire the constraint of a given date.
    # Pad 2 months backward and forward for constraint.
    def __getDateConstraints(self, date):
        minDate = datetime(2001, 1, 1); # Database beginning date
        originalDate = datetime.strptime(date, "%m/%d/%Y");

        # Define the lower-bound date
        lbMonth = originalDate.month - 2;
        if (lbMonth <= 0): # Case: Wrap to previous year
            lbMonth += 12;
            lbYear = originalDate.year - 1;
        else: # Case: Still on current year
            lbYear = originalDate.year;

        # Construct lower-bound date
        try:
            lowerBoundDate = originalDate.replace(year=lbYear, month=lbMonth);
        except ValueError: # Catch potential error i.e. feb. 30 not existing
            lowerBoundDate = originalDate.replace(year=lbYear, month=lbMonth, day=1);

        # Ensure the new date does not go below the minimum date
        if (lowerBoundDate < minDate):
            lowerBoundDate = minDate;

        
        # Define the upper-bound date
        ubMonth = originalDate.month + 2;
        if (ubMonth > 12): # Case: Wrap to next year
            ubMonth -= 12;
            ubYear = originalDate.year + 1;
        else: # Case: Still on current year
            ubYear = originalDate.year;

        # Construct upper-bound date
        try:
            upperBoundDate = originalDate.replace(year=ubYear, month=ubMonth);
        except ValueError: # Catch potential error i.e. feb. 30 not existing
            upperBoundDate = originalDate.replace(year=ubYear, month=ubMonth + 1, day=1);

        return [lowerBoundDate, upperBoundDate];

    """
        - Attempt to get the list of CIKs for the merging companies
        - If no CIK is found, return None
            - This will indicate for a more broad search of all the files
    """
    def __getCIKS(self, searchCompany, pairCompany, dateLB, dateUB, formTypes):
        restructName = searchCompany.replace(" ", "%20");
        
        url = f"https://efts.sec.gov/LATEST/search-index?q={restructName}&dateRange=custom&category=custom&startdt={dateLB}&enddt={dateUB}&forms={formTypes}";

        # Create a request that mimics browser activity
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
            "Referer": "https://www.sec.gov/"
        }

        # Request the search query & acquire the DOM elements
        response = requests.get(url, headers=headers);
        if (response.status_code != 200):
            print("FATAL: getDocumentJson response yielded an error!");
            sys.exit(response.status_code);
        
        data = response.json();
        totalValue = data["hits"]["total"]["value"];

        if (totalValue <= 0):
            return None;
        
        # Formulate the list of entities for CIK lookup
        entityList = [];
        for entities in data["aggregations"]["entity_filter"]["buckets"]:
            entityList.append(entities["key"]);

        # Acquire the CIK for the given company using fuzzy matching techniques
        threshold = 90;
        filteredMatch = [
            entity for entity in entityList if fuzz.partial_ratio(pairCompany.lower(), entity.lower()) > threshold
        ];
        
        # Extract the CIK from the filtered match
        cikList = [];
        for entity in filteredMatch:
            cikList.append(re.search(r'\(CIK (\d+)\)', entity).group(1));

        return cikList if cikList else None;

    # Acquire all the json documents for the given companies with **CIK filter enabled**
    def __getCIKDocumentJson(self, searchCompany, pairCompany, dateLB, dateUB, formTypes):
        # Remove parantheses content from the company names
        searchCompany = re.sub(r'\(.*\)', '', searchCompany).strip();
        pairCompany = re.sub(r'\(.*\)', '', pairCompany).strip();

        # We will try and acquire the cikList for the first company;
        # If the cikList is None, we will try and acquire the cikList for the second company.
        cikList = self.__getCIKS(searchCompany, pairCompany, dateLB, dateUB, formTypes);
        if (cikList == None):
            cikList = self.__getCIKS(pairCompany, searchCompany, dateLB, dateUB, formTypes);
        
        if (cikList == None):
            return None;

        """
            - Fetch data for each CIK concurrently
            - We do not need to verify if the hit returns nothing as if the entity is not found,
                the CIK will not be present in the list.
        """
        restructName = searchCompany.replace(" ", "%20");

        urls = [f"https://efts.sec.gov/LATEST/search-index?q={restructName}&dateRange=custom&category=custom&startdt={dateLB}&enddt={dateUB}&forms={formTypes}&filter_ciks={cik}" for cik in cikList];
        
        # Create a request that mimics browser activity
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
            "Referer": "https://www.sec.gov/"
        }

        # Fetch the json data for each CIK
        if len(urls) == 1: # Case: Single URL; no threads required
            response = requests.get(urls[0], headers=headers);
            if (response.status_code != 200):
                print("FATAL: getDocumentJson response yielded an error!");
                sys.exit(response.status_code);
            
            result = response.json();
            mergedHits = result["hits"]["hits"] if result and "hits" in result and "hits" in result["hits"] else [];
        else: # Case: Multiple URLs; use threads for concurrent fetching
            with ThreadPoolExecutor(max_workers=self.threadCount) as executor:
                results = list(executor.map(lambda url: requests.get(url, headers=headers), urls));
            
            # Merge the results into a single list
            mergedHits = [];
            for response in results:
                if (response.status_code != 200):
                    print("FATAL: getDocumentJson response yielded an error!");
                    sys.exit(response.status_code);

                result = response.json();
                if result and "hits" in result and "hits" in result["hits"]:
                    mergedHits.extend(result["hits"]["hits"]);

        return mergedHits if mergedHits else None;

    """
        - No documents were found associated with the CIKs.
        - We will let fuzzy match determine if the company is present in the document.
            - Return a truncated list of documents for both companies without the cik filter.
        - Basically throwing a dart at the board and hoping it hits the target if no cik filtering is found.
    """
    def __getDocumentJson(self, searchCompany, pairCompany, dateLB, dateUB, formTypes):
        # Remove parantheses content from the company names
        searchCompany = re.sub(r'\(.*\)', '', searchCompany).strip();
        pairCompany = re.sub(r'\(.*\)', '', pairCompany).strip();

        restructSearch = searchCompany.replace(" ", "%20");
        restructPair = pairCompany.replace(" ", "%20");

        urls = [
            f"https://efts.sec.gov/LATEST/search-index?q={restructSearch}&dateRange=custom&category=custom&startdt={dateLB}&enddt={dateUB}&forms={formTypes}",
            f"https://efts.sec.gov/LATEST/search-index?q={restructPair}&dateRange=custom&category=custom&startdt={dateLB}&enddt={dateUB}&forms={formTypes}"
        ];
        
        # Create a request that mimics browser activity
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
            "Referer": "https://www.sec.gov/"
        }

        # Fetch the json data for each company using threads for concurrent fetching
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda url: requests.get(url, headers=headers), urls));
        
        # Merge the results into a single list
        mergedHits = [];
        for response in results:
            if (response.status_code != 200):
                print("FATAL: getDocumentJson response yielded an error!");
                sys.exit(response.status_code);

            result = response.json();
            if result and "hits" in result and "hits" in result["hits"]:
                mergedHits.extend(result["hits"]["hits"]);

        return mergedHits if mergedHits else None;

    # Formulate the source document links from the search result json
    def __getSourceLinks(self, documentJson):
        # Formulate all source document file links
        sourceLinks = [];
        seenLinks = set(); # Prevent duplicates

        # Iterate through each json object and construct the source document file links
        for document in documentJson:
            try:
                # Get the CIK id or if there is multiple, then acquire the last one
                validatedCik = None;
                ciks = document["_source"]["ciks"];
                if ciks:
                    validatedCik = ciks[-1].lstrip('0');
            
                # Acquire normal adsh & adsh without the "-" character
                adsh = document["_source"]["adsh"];
                truncatedADSH = document["_source"]["adsh"].replace("-", "");

                # Add non-duplicate urls
                url = f"https://www.sec.gov/Archives/edgar/data/{validatedCik}/{truncatedADSH}/{adsh}.txt"
                if url not in seenLinks:
                    seenLinks.add(url)
                    sourceLinks.append(url)
                
            except KeyError as e:
                Logger.logMessage(f"[{Logger.get_current_timestamp()}] [-] Missing key in document: {e}, result: {document}");
                continue; # Skip the document if there is a missing key; logged for further investigation

        return sourceLinks;

    def runCrawler(self, startIndex: int = None, endIndex: int = None, index: int = None):
        # If index is provided, override startIndex and endIndex
        if index is not None:
            self.__startIndex = index;
            self.__endIndex = index + 1;
        else:
            # Ensure startIndex and endIndex are set properly
            self.__startIndex = startIndex if startIndex is not None else 0;
            self.__endIndex = endIndex if endIndex is not None else self.__startIndex + 1;

        # Create the thread pool for efficiently allocate threads via executor
        with ThreadPoolExecutor(max_workers=self.threadCount) as executor:
            # Initiate the Processor to clean & analzye the documents
            processor = Processor(self.assistant, self.nlp, self.threadCount, self.startPhrases, executor);
            
            # Stores all asynchronous results from Assistant and load them at the end.
            # This preserves the operation of multi-threading.
            futures = [];
            for mainIndex in tqdm(
                range(self.__startIndex, self.__endIndex),
                desc="\033[35mProcessing\033[0m",
                unit="items",
                ncols=80,
                bar_format="\033[92m{desc}: {percentage:3.0f}%|\033[92m{bar}\033[0m| {n_fmt}/{total_fmt} [elapsed: {elapsed}]\n"
            ):
                print("Processing index: ", mainIndex, "; Companies: ", self.companyAList[mainIndex], " & ", self.companyBList[mainIndex]);

                # Construct the constraint of a given date & prep for url-parsing
                constraintDates = self.__getDateConstraints(self.filedDate[mainIndex]);
                lbDate, ubDate = constraintDates;
                restructLB = f"{lbDate.year}-{lbDate.month:02}-{lbDate.day:02}";
                restructUB = f"{ubDate.year}-{ubDate.month:02}-{ubDate.day:02}";
                restructForms = "%2C".join(self.__formTypes).replace(" ", "%20");

                # Find the documents with CIK filtering
                results = self.__getCIKDocumentJson(self.companyAList[mainIndex], self.companyBList[mainIndex], restructLB, restructUB, restructForms);
                if (results == None): # Acquire all documents within our guess
                    results = self.__getDocumentJson(self.companyAList[mainIndex], self.companyBList[mainIndex], restructLB, restructUB, restructForms);

                # No documents found for our 2 companies
                if (results == None):
                    Logger.logMessage(f"[{Logger.get_current_timestamp()}] [-] No document found for: {self.companyAList[mainIndex]} & {self.companyBList[mainIndex]}");
                    continue;
                
                # Extract the source document links
                sourceLinks = self.__getSourceLinks(results);

                # Filter the documents and keep the ones with the existence of both company names
                companyNames = [self.companyAList[mainIndex], self.companyBList[mainIndex]];
                documents = processor.getDocuments(sourceLinks, companyNames);
                if not documents:
                    Logger.logMessage(f"[{Logger.get_current_timestamp()}] [-] No relevant document found for: {self.companyAList[mainIndex]} & {self.companyBList[mainIndex]}");
                    continue;

                future = processor.locateSection(documents, companyNames, mainIndex);
                if future:
                    futures.append((mainIndex, future));

                # Process the documents from the source links and save to be
                # processed after mainIndex loop finishes
                # processResult = processor.locateSection(sourceLinks, companyNames, mainIndex);
                # print(type(processResult))
                # if future:
                #     futures.append((mainIndex, future));
            
            print("Waiting for all asynchronous threads to finish...\n");

            # Wait for all OpenAI messages to complete before processing
            wait([future for _, future in futures], return_when=ALL_COMPLETED);

            # Now process the results to extract the initiator from the Assistant output response
            with open("output.csv", mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file);
                if file.tell() == 0:
                    writer.writerow(["INDEX", "DATE", "TMANAMES", "AMANAMES", "INITIATOR"]);

                for mainIndex, future in tqdm(
                    futures,
                    total=len(futures),
                    desc="\033[33mWriting to CSV\033[0m",
                    unit="items",
                    ncols=80,
                    bar_format="\033[92m{desc}: {percentage:3.0f}%|\033[92m{bar}\033[0m| {n_fmt}/{total_fmt} [elapsed: {elapsed}]\n"
                ):
                    try:
                        result = future.result();
                        if result is None:
                            continue;
                        
                        # Extract the initiator from the specified format: [company name]
                        match = re.search(r"\[(.*?)\]", result);
                        initiator = match.group(1) if match else "Unknown";
                        
                        # Write to the output CSV
                        writer.writerow([f"index_{mainIndex}", self.filedDate[mainIndex], self.companyAList[mainIndex], self.companyBList[mainIndex], initiator]);
                    except Exception as e:
                        Logger.logMessage(f"[{Logger.get_current_timestamp()}] [-] Error processing future for index {mainIndex}: {e}");
                        self.assistant.clearVectorStores();

            # Clean up the vector store at the end as we can't clear while in parallel processing
            self.assistant.clearVectorStores();