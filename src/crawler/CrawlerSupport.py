from concurrent.futures import ThreadPoolExecutor
import os
from datetime import datetime
import sys
import re
from rapidfuzz import fuzz
import requests
import gc
import time

from src.dependencies.RateLimiter import RateLimiter
from src.utils.Logger import Logger
from src.crawler.Processor import Processor

from src.dependencies.config import FORM_TYPES, MAX_NUM_OF_THREADS

"""
    - This object houses static helper methods for the Crawler.
    - Intermediator to multiprocessing tasks.
"""
class CrawlerSupport:
    @staticmethod
    def rate_limited_get(url: str, headers: any, rate_limiter_resources: dict[str, any]):
        """Wrapper for GET request with rate limiting"""
        RateLimiter.wait(rate_limiter_resources);
        response = requests.get(url, headers=headers);
        return response;
    
    @staticmethod
    def get_date_constraints(date: str, margin: int = 2) -> list[datetime]:
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

    @staticmethod
    def get_ciks(
        search_company: str, 
        pair_company: str, 
        date_LB: str, 
        date_UB: str, 
        form_types: list[str],
        rate_limiter_resources: dict[str, any] 
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
            rate_limiter_resources : dict[str, any]
                Request wait limiter to stop flooding.
                
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
        response = CrawlerSupport.rate_limited_get(url, headers, rate_limiter_resources)
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

    @staticmethod
    def get_cik_document_json(
        search_company: str, 
        pair_company: str, 
        date_LB: str, 
        date_UB: str, 
        form_types: list[str], 
        rate_limiter_resources: dict[str, any] 
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
                The defined maximum number of threads per thread pool.
            rate_limiter_resources : dict[str, any]
                Request wait limiter to stop flooding.
                
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
        cik_list = CrawlerSupport.get_ciks(search_company, pair_company, date_LB, date_UB, form_types, rate_limiter_resources);
        if (cik_list == None):
            cik_list = CrawlerSupport.get_ciks(pair_company, search_company, date_LB, date_UB, form_types, rate_limiter_resources);
        
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
            response = CrawlerSupport.rate_limited_get(urls[0], headers, rate_limiter_resources);
            if (response.status_code != 200):
                print("FATAL: getDocumentJson response yielded an error!");
                sys.exit(response.status_code);
            
            result = response.json();
            merged_hits = result["hits"]["hits"] if result and "hits" in result and "hits" in result["hits"] else [];
        else: # Case: Multiple URLs; use threads for concurrent fetching
            with ThreadPoolExecutor(max_workers=MAX_NUM_OF_THREADS) as thread_pool:
                results = list(
                    thread_pool.map(lambda url: CrawlerSupport.rate_limited_get(url, headers, rate_limiter_resources), urls)
                );
                
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

    @staticmethod
    def get_document_json(
        search_company: str, 
        pair_company: str, 
        date_LB: str, 
        date_UB: str, 
        form_types: list[str],
        rate_limiter_resources: dict[str, any] 
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
            rate_limiter_resources : dict[str, any]
                Request wait limiter to stop flooding.
                
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
        with ThreadPoolExecutor(max_workers=MAX_NUM_OF_THREADS) as thread_pool:
            results = list(
                thread_pool.map(lambda url: CrawlerSupport.rate_limited_get(url, headers, rate_limiter_resources), urls)
            )
        
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

    @staticmethod
    def get_source_links(document_jsons: list[dict]) -> list[str]:
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
                url = f"https://www.sec.gov/Archives/edgar/data/{validated_cik}/{truncated_ADSH}/{adsh}.txt";
                if url not in seen_links:
                    seen_links.add(url);
                    source_links.append(url);
                
            except KeyError as e:
                Logger.logMessage(f"[-] Missing key in document: {e}, result: {document}");
                continue; # Skip the document if there is a missing key; logged for further investigation

        return source_links;

    @staticmethod
    def process_single_job(
        job_data: tuple[int, str, str, str], 
        date_margin: int,
        rate_limiter_resources: dict[str, any] 
    ):
        main_index, company_A, company_B, announcement_date = job_data;
        print("Processing index: ", main_index, "; Companies: ", company_A, " & ", company_B);

        # Construct document file name & construct the folder constraint
        company_names = [company_A, company_B];
        format_doc_name = f"{main_index}_{company_names[0].replace(' ', '_')}_&_{company_names[1].replace(' ', '_')}";

        batch_start = (main_index // 100) * 100;
        batch_end = batch_start + 99;
        
        # Check if the file exists
        file_path = f"./DataSet/{batch_start}-{batch_end}/{format_doc_name}.txt";
        if os.path.isfile(file_path):
            print("Skipping: Document already exist...");
            return None;
    
        # Construct the constraint of a given date & prep for url-parsing
        kwargs = {'date': announcement_date};
        if date_margin is not None:
            kwargs['margin'] = date_margin;
        
        constraint_dates = CrawlerSupport.get_date_constraints(**kwargs);
        lb_Date, ub_Date = constraint_dates;
        restruct_LB = f"{lb_Date.year}-{lb_Date.month:02}-{lb_Date.day:02}";
        restruct_UB = f"{ub_Date.year}-{ub_Date.month:02}-{ub_Date.day:02}";
        restruct_forms = "%2C".join(FORM_TYPES).replace(" ", "%20");

        # Find the documents with CIK filtering
        results = CrawlerSupport.get_cik_document_json(
            company_A, 
            company_B, 
            restruct_LB, 
            restruct_UB, 
            restruct_forms, 
            rate_limiter_resources
        );
        if (results == None): # Acquire all documents within our guess
            results = CrawlerSupport.get_document_json(
                company_A, 
                company_B, 
                restruct_LB, 
                restruct_UB, 
                restruct_forms,
                rate_limiter_resources
            );

        # No documents found for our 2 companies
        if (results == None):
            Logger.logMessage(f"[-] No document found for: {company_A} & {company_B}");
            return None;
    
        # Extract the source document links
        source_links = CrawlerSupport.get_source_links(results);

        # Filter the documents and keep the ones with the existence of both company names
        company_names = [company_A, company_B];
        documents = Processor.getDocuments(source_links, company_names, rate_limiter_resources);

        # Retry if no documents found and any company name has a hyphen (Edge case)
        if not documents and any("-" in name for name in company_names):
            modified_names = [
                name.replace('-', ' ') if '-' in name else name
                for name in company_names
            ];
            documents = Processor.getDocuments(source_links, modified_names, rate_limiter_resources);

        # No documents found, including the edge case
        if not documents:
            Logger.logMessage(
                f"[-] No relevant document found for index {main_index}: {company_A} & {company_B}"
            );
            return None;
        
        print(f"Number of documents: {len(documents)}");

        # Acquire the specific document with the "Background of the Merger" section
        doc_url = Processor.locateDocument(documents, company_names, main_index);
        if doc_url is None:
            Logger.logMessage(
                f"[-] Confirmed no background section found for index {main_index}: {company_names[0]} & {company_names[1]}."
            );
            Logger.logMessage(f"\tDumping its document links:", time_stamp=False);
            for doc in documents:
                Logger.logMessage(f"\t\t{doc.getUrl()}", time_stamp=False);
            return None;
        
        # Save the document for writing at the end
        return (main_index, doc_url);
