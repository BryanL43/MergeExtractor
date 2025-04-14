from datetime import datetime
import re
import requests
import sys
from concurrent.futures import ThreadPoolExecutor
from rapidfuzz import fuzz
import csv
from tqdm import tqdm
import time
import gc
from spacy.language import Language
import os

from Logger import Logger
from BackupAssistant import BackupAssistant
from Processor import Processor

class Crawler:
    def __init__(
            self,
            filed_date: list[str],
            company_A_list: list[str],
            company_B_list: list[str],
            start_phrases: list[str],
            thread_pool: ThreadPoolExecutor,
            nlp: Language,
            assistant: BackupAssistant
        ):
        
        self.filed_date = filed_date;
        self.company_A_list = company_A_list;
        self.company_B_list = company_B_list;
        self.start_phrases = start_phrases;
        self.thread_pool = thread_pool;
        self.nlp = nlp;
        self.assistant = assistant;

        self.__form_types = ["PREM14A", "S-4", "SC 14D9", "SC TO-T"];

        # Instantiate the Processor to clean & analzye documents
        self._processor = Processor(self.assistant, self.nlp, self.start_phrases, self.thread_pool);

    # Acquire the constraint of a given date.
    # Pad 2 months backward and forward for constraint.
    def __get_date_constraints(self, date: str, margin: int = 2) -> list[datetime]:
        """
            Acquire the constraint of a given date.
            Pad margin amount of months backward and forward for constraint.

            Parameters
            ----------
            date : str
                The date to acquire the constraints of, in the format of "%m/%d/%Y"
            margin : int, optional
                The number of months to pad, by default 2

            Returns
            -------
            list : datetime
                A list of two datetime objects, representing the lower-bound and upper-bound dates
        """
        min_date = datetime(2001, 1, 1); # Database beginning date
        original_date = datetime.strptime(date, "%m/%d/%Y");

        # Define the lower-bound date
        lb_month = original_date.month - margin;
        if (lb_month <= 0): # Case: Wrap to previous year
            lb_month += 12;
            lb_year = original_date.year - 1;
        else: # Case: Still on current year
            lb_year = original_date.year;

        # Construct lower-bound date
        try:
            lower_bound_date = original_date.replace(year=lb_year, month=lb_month);
        except ValueError: # Catch potential error i.e. feb. 30 not existing
            lower_bound_date = original_date.replace(year=lb_year, month=lb_month, day=1);

        # Ensure the new date does not go below the minimum date
        if (lower_bound_date < min_date):
            lower_bound_date = min_date;

        
        # Define the upper-bound date
        ub_month = original_date.month + margin;
        if (ub_month > 12): # Case: Wrap to next year
            ub_month -= 12;
            ub_year = original_date.year + 1;
        else: # Case: Still on current year
            ub_year = original_date.year;

        # Construct upper-bound date
        try:
            upper_bound_date = original_date.replace(year=ub_year, month=ub_month);
        except ValueError: # Catch potential error i.e. feb. 30 not existing
            upper_bound_date = original_date.replace(year=ub_year, month=ub_month + 1, day=1);

        return [lower_bound_date, upper_bound_date];

    """
        - Attempt to get the list of CIKs for the merging companies
        - If no CIK is found, return None
            - This will indicate for a more broad search of all the files
    """
    def __get_ciks(
            self, 
            search_company: str, 
            pair_company: str, 
            date_LB: str, 
            date_UB: str, 
            form_types: list[str]
        ) -> list[int] | None:
        """
            Attempt to get the list of CIKs ID for the merging companies.

            Parameters
            ----------
            search_company : str
                The initial company to search for.
            pair_company : str
                The associating company to search for via filter.
            date_LB : str
                The lower-bound date, in the format of "%Y-M-%D"
            date_UB : str
                The upper-bound date, in the format of "%Y-M-%D"
            form_types : list[str]
                The form types to search for.
                
            Returns
            -------
            list : int
                A list of the CIKs for the merging companies.
            None
                if we cannot acquire anything.
        """
        restruct_name = search_company.replace(" ", "%20");
        
        url = f"https://efts.sec.gov/LATEST/search-index?q={restruct_name}&dateRange=custom&category=custom&startdt={date_LB}&enddt={date_UB}&forms={form_types}";

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
        total_value = data["hits"]["total"]["value"];

        if (total_value <= 0):
            return None;
        
        # Formulate the list of entities for CIK lookup
        entity_list = [];
        for entities in data["aggregations"]["entity_filter"]["buckets"]:
            entity_list.append(entities["key"]);

        # Acquire the CIK for the given company using fuzzy matching techniques
        threshold = 90;
        filtered_match = [
            entity for entity in entity_list if fuzz.partial_ratio(pair_company.lower(), entity.lower()) > threshold
        ];
        
        # Extract the CIK from the filtered match
        cik_list = [];
        for entity in filtered_match:
            cik_list.append(re.search(r'\(CIK (\d+)\)', entity).group(1));

        return cik_list if cik_list else None;

    def __get_cik_document_json(
            self, 
            search_company: str, 
            pair_company: str, 
            date_LB: str, 
            date_UB: str, 
            form_types: list[str]
        ) -> list[dict] | None:
        """
            Acquire all the json documents for the given companies with **CIK filter enabled**.

            Parameters
            ----------
            search_company : str
                The initial company to search for.
            pair_company : str
                The associating company to search for via filter.
            date_LB : str
                The lower-bound date, in the format of "%Y-M-%D"
            date_UB : str
                The upper-bound date, in the format of "%Y-M-%D"
            form_types : list[str]
                The form types to search for.
                
            Returns
            -------
            list : dict
                A list of the document jsons for the merging companies.
            None
                if we cannot acquire anything.
        """
        # Remove parantheses content from the company names
        search_company = re.sub(r'\(.*\)', '', search_company).strip();
        pair_company = re.sub(r'\(.*\)', '', pair_company).strip();

        # We will try and acquire the cik_list for the first company;
        # If the cik_list is None, we will try and acquire the cik_list for the second company.
        cik_list = self.__get_ciks(search_company, pair_company, date_LB, date_UB, form_types);
        if (cik_list == None):
            cik_list = self.__get_ciks(pair_company, search_company, date_LB, date_UB, form_types);
        
        if (cik_list == None):
            return None;

        """
            - Fetch data for each CIK concurrently
            - We do not need to verify if the hit returns nothing as if the entity is not found,
                the CIK will not be present in the list.
        """
        restruct_name = search_company.replace(" ", "%20");

        urls = [
            f"https://efts.sec.gov/LATEST/search-index?q={restruct_name}&dateRange=custom&category=custom&startdt={date_LB}&enddt={date_UB}&forms={form_types}&filter_ciks={cik}"
            for cik in cik_list
        ];
        
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
            merged_hits = result["hits"]["hits"] if result and "hits" in result and "hits" in result["hits"] else [];
        else: # Case: Multiple URLs; use threads for concurrent fetching
            results = list(self.thread_pool.map(lambda url: requests.get(url, headers=headers), urls));
            
            # Merge the results into a single list
            merged_hits = [];
            for response in results:
                if (response.status_code != 200):
                    print("FATAL: getDocumentJson response yielded an error!");
                    sys.exit(response.status_code);

                result = response.json();
                if result and "hits" in result and "hits" in result["hits"]:
                    merged_hits.extend(result["hits"]["hits"]);

        return merged_hits if merged_hits else None;

    def __get_document_json(
            self, 
            search_company: str, 
            pair_company: str, 
            date_LB: str, 
            date_UB: str, 
            form_types: list[str]
        ) -> list[dict] | None:
        """
            Acquire all the json documents for the given companies with **CIK filter disabled**.
            Basically throwing a dart at the board and hoping it hits the target if no cik filtering is found.

            Parameters
            ----------
            search_company : str
                The initial company to search for.
            pair_company : str
                The associating company to search for via filter.
            date_LB : str
                The lower-bound date, in the format of "%Y-M-%D"
            date_UB : str
                The upper-bound date, in the format of "%Y-M-%D"
            form_types : list[str]
                The form types to search for.
                
            Returns
            -------
            list : dict
                A list of the document jsons for the merging companies.
            None
                if we cannot acquire anything.
        """
        # Remove parantheses content from the company names
        search_company = re.sub(r'\(.*\)', '', search_company).strip();
        pair_company = re.sub(r'\(.*\)', '', pair_company).strip();

        restruct_search = search_company.replace(" ", "%20");
        restruct_pair = pair_company.replace(" ", "%20");

        urls = [
            f"https://efts.sec.gov/LATEST/search-index?q={restruct_search}&dateRange=custom&category=custom&startdt={date_LB}&enddt={date_UB}&forms={form_types}",
            f"https://efts.sec.gov/LATEST/search-index?q={restruct_pair}&dateRange=custom&category=custom&startdt={date_LB}&enddt={date_UB}&forms={form_types}"
        ];
        
        # Create a request that mimics browser activity
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
            "Referer": "https://www.sec.gov/"
        }

        # Fetch the json data for each company using threads for concurrent fetching
        results = list(self.thread_pool.map(lambda url: requests.get(url, headers=headers), urls));
        
        # Merge the results into a single list
        merged_hits = [];
        for response in results:
            if (response.status_code != 200):
                print("FATAL: getDocumentJson response yielded an error!");
                sys.exit(response.status_code);

            result = response.json();
            if result and "hits" in result and "hits" in result["hits"]:
                merged_hits.extend(result["hits"]["hits"]);

        return merged_hits if merged_hits else None;

    def __get_source_links(self, document_jsons: list[dict]) -> list[str]:
        """
            Formulate the source document links from the search result json.

            Parameters
            ----------
            document_jsons : list[dict]
                A list of the document jsons for the merging companies.
                
            Returns
            -------
            list : str
                A list of document source links.
        """
        # Formulate all source document file links
        source_links = [];
        seen_links = set(); # Prevent duplicates

        # Iterate through each json object and construct the source document file links
        for document in document_jsons:
            try:
                # Get the CIK id or if there is multiple, then acquire the last one
                validated_cik = None;
                ciks = document["_source"]["ciks"];
                if ciks:
                    validated_cik = ciks[-1].lstrip('0');
            
                # Acquire normal adsh & adsh without the "-" character
                adsh = document["_source"]["adsh"];
                truncated_ADSH = document["_source"]["adsh"].replace("-", "");

                # Add non-duplicate urls
                url = f"https://www.sec.gov/Archives/edgar/data/{validated_cik}/{truncated_ADSH}/{adsh}.txt"
                if url not in seen_links:
                    seen_links.add(url)
                    source_links.append(url)
                
            except KeyError as e:
                Logger.logMessage(f"[-] Missing key in document: {e}, result: {document}");
                continue; # Skip the document if there is a missing key; logged for further investigation

        return source_links;

    def __write_output_urls(self, acquired_documents: list[tuple[int, str]]):
        """
            Writes the output urls to the csv file.

            Parameters
            ----------
            acquired_documents : list[tuple[int, str]]
                A list of the acquired document's url in the tuple of (associated index, url).
        """
        print("Writing results to CSV...");
        with open("output.csv", mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file);
            if file.tell() == 0:
                writer.writerow(["INDEX", "ANNOUNCEMENT_DATE", "TMANAMES", "AMANAMES", "URL"]);

            for main_index, url in acquired_documents:
                try:
                    # Write to the output CSV
                    writer.writerow(
                        [main_index, self.filed_date[main_index], self.company_A_list[main_index], self.company_B_list[main_index], url]
                    );
                except Exception as e:
                    Logger.logMessage(f"[-] Error writing to output for index {main_index}: {e}");

    def __resetResources(self):
        """Garbage collection"""
        gc.collect(); # CPU flush
        time.sleep(2);

    def runCrawler(self, start_index: int = None, end_index: int = None, index: int = None, date_margin: int = None):
        """
            Main operation for the US SEC EDGAR database crawler.

            Parameters
            ----------
            start_index : int
                The start index of the truncated data.
            end_index : int
                The end index of the truncated data.
            index: int, Optional
                Single case index of the truncated data.
            date_margin: int, Optional
                The margin padding for the date range to search. Pads both LHS and RHS by margin starting at the given date.
                Example: If the date is 5/16/2001 and margin is 2, then the date range will be 3/16/2001 to 7/16/2001.
            
            Raises
            ----------
                ValueError: If there is a missing index.
        """
        # If index is provided, override startIndex and endIndex
        if index is not None:
            self.__start_index = index;
            self.__end_index = index + 1;
        else:
            # Ensure startIndex and endIndex are set properly
            if start_index is None or end_index is None:
                raise ValueError("start_index and end_index must be provided if index is not provided.");

            self.__start_index = start_index;
            self.__end_index = end_index + 1;
            
        acquired_documents = []; # Stores all successfully located documents to write at the end
        for main_index in tqdm(
            range(self.__start_index, self.__end_index),
            desc="\033[35mProcessing\033[0m",
            unit="items",
            ncols=80,
            bar_format="\033[92m{desc}: {percentage:3.0f}%|\033[92m{bar}\033[0m| {n_fmt}/{total_fmt} [elapsed: {elapsed}]\n"
        ):
            print("Processing index: ", main_index, "; Companies: ", self.company_A_list[main_index], " & ", self.company_B_list[main_index]);

            # Construct document file name & construct the folder constraint
            company_names = [self.company_A_list[main_index], self.company_B_list[main_index]];
            format_doc_name = f"{main_index}_{company_names[0].replace(' ', '_')}_&_{company_names[1].replace(' ', '_')}";

            batch_start = (main_index // 100) * 100;
            batch_end = batch_start + 99;
            
            # Check if the file exists
            file_path = f"./DataSet/{batch_start}-{batch_end}/{format_doc_name}.txt";
            if os.path.isfile(file_path):
                print("Skipping: Document already exist exist...");
                continue;

            # Construct the constraint of a given date & prep for url-parsing
            kwargs = {'date': self.filed_date[main_index]};
            if date_margin is not None:
                kwargs['margin'] = date_margin;

            constraint_dates = self.__get_date_constraints(**kwargs);
            lb_Date, ub_Date = constraint_dates;
            restruct_LB = f"{lb_Date.year}-{lb_Date.month:02}-{lb_Date.day:02}";
            restruct_UB = f"{ub_Date.year}-{ub_Date.month:02}-{ub_Date.day:02}";
            restruct_forms = "%2C".join(self.__form_types).replace(" ", "%20");

            # Find the documents with CIK filtering
            results = self.__get_cik_document_json(self.company_A_list[main_index], self.company_B_list[main_index], restruct_LB, restruct_UB, restruct_forms);
            if (results == None): # Acquire all documents within our guess
                results = self.__get_document_json(self.company_A_list[main_index], self.company_B_list[main_index], restruct_LB, restruct_UB, restruct_forms);

            # No documents found for our 2 companies
            if (results == None):
                Logger.logMessage(f"[-] No document found for: {self.company_A_list[main_index]} & {self.company_B_list[main_index]}");
                self.__resetResources();
                continue;
            
            # Extract the source document links
            source_links = self.__get_source_links(results);

            # Filter the documents and keep the ones with the existence of both company names
            company_names = [self.company_A_list[main_index], self.company_B_list[main_index]];
            documents = self._processor.getDocuments(source_links, company_names);

            # Retry if no documents found and any company name has a hyphen (Edge case)
            if not documents and any("-" in name for name in company_names):
                modified_names = [
                    name.replace('-', ' ') if '-' in name else name
                    for name in company_names
                ];
                documents = self._processor.getDocuments(source_links, modified_names);

            # No documents found, including the edge case
            if not documents:
                Logger.logMessage(
                    f"[-] No relevant document found for index {main_index}: {self.company_A_list[main_index]} & {self.company_B_list[main_index]}"
                );
                self.__resetResources();
                continue;
            
            print(f"Number of documents: {len(documents)}");

            # Acquire the specific document with the "Background of the Merger" section
            doc_url = self._processor.locateDocument(documents, company_names, main_index);
            if doc_url is None:
                Logger.logMessage(
                    f"[-] Confirmed no background section found for index {main_index}: {company_names[0]} & {company_names[1]}."
                );
                Logger.logMessage(f"\tDumping its document links:", time_stamp=False);
                for doc in documents:
                    Logger.logMessage(f"\t\t{doc.getUrl()}", time_stamp=False);
                self.__resetResources();
                continue;
            
            # Save the document for writing at the end
            acquired_documents.append((main_index, doc_url));

            self.__resetResources();

        self.__write_output_urls(acquired_documents);

        # Clean up the vector store at the end as we can't clear while in parallel processing
        self.assistant.clearVectorStores();
