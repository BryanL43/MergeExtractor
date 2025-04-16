import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait, ALL_COMPLETED, CancelledError
from multiprocessing import Manager, get_context
from threading import Event
import sys
import requests
import os
import shutil
import random
import time
import unicodedata
from lxml import etree
from spacy.language import Language

from BackupAssistant import BackupAssistant
from Logger import Logger
from Document import Document
# from ChunkProcessor import ChunkProcessor
from RateLimiter import RateLimiter

TEMP_DIRECTORY = "merge_extractor_temp";

"""
    - This object handles processing the document by:
        - Cleaning the data
        - Verifying the existence of both companies are present
        - Extracting the background section
    - Static methods to accomodate multiprocessing task
"""
class Processor:
    @staticmethod
    def extract_all_but_last_word(company_name: str) -> str:
        """
            Given a company name, drop the last word and merge domain-like terms.

            Parameters
            ----------
            company_name : str
                The company name to be processed.

            Returns
            -------
            str
                The processed company name.
        """
        clean_name = re.sub(r"\(.*?\)", "", company_name);  # Remove parentheses content
        words = re.split(r"[\s\_]+", clean_name.strip());  # Split by space or underscore

        # Domain-like terms to merge
        merge_words = {"net", "com", "org", "co"};

        # Merge domain-like words
        for i in range(len(words) - 1):
            if words[i].lower() in merge_words:
                words[i] = words[i] + "." + words[i + 1];
                words.pop(i + 1);
                break;

        if len(words) > 1:
            if words[-2] == "&":
                words = words[:-2]; # Remove both "&" and the last word
            else:
                words = words[:-1];  # Remove only the last word

        return " ".join(words);

    @staticmethod
    def load_file_from_url(url: str, rate_limiter_resources: dict[str, any]) -> str:
        """
            Acquire the document's text from the given url.

            Parameters
            ----------
            url : str
                The document source url.
            rate_limiter_resources : dict[str, any]
                Request wait limiter to stop flooding.

            Returns
            -------
            str
                The document's text content.
        """
        # Create a request that mimics browser activity
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }

        RateLimiter.wait(rate_limiter_resources);
        response = requests.get(url, headers=headers);
        if (response.text):
            return response.text;
        else:
            print(f"FATAL: Failed to load document via url. Err_Code: {response.status_code}");
            sys.exit(response.status_code);
    
    @staticmethod
    def preprocess_text(content: str) -> str:
        """
            Initial document text sanitizing.
            Parse the content as utf-8, roughly remove page numbers, and remove large consecutive newlines.

            Parameters
            ----------
            content : str
                The document's text.

            Returns
            -------
            str
                The sanitized document's text content.
        """
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

    @staticmethod
    def normalize_text(text: str) -> str:
        """
            Remove table of contents references and normalize the text

            Parameters
            ----------
            text : str
                The document's text.

            Returns
            -------
            str
                The normalized text.
        """
        text = unicodedata.normalize("NFKC", text);  # Normalize Unicode
        text = text.encode("ascii", "ignore").decode("ascii");
        cleanedText = re.sub(r'\btable\s*of\s*contents?\b|\btableofcontents?\b', '', text, flags=re.IGNORECASE);
        cleanedText = re.sub(r'(?i)table\s*of\s*contents?|tableofcontents?', '', cleanedText);

        return cleanedText.strip();

    @staticmethod
    def check_companies_in_document(url: str, company_names: list[str], rate_limiter_resources: dict[str, any]) -> tuple[str, bool]:
        """
            Validate the presence of both companys' name in the document.

            Parameters
            ----------
            url : str
                The document source url.
            company_names : list[str]
                The list of sanitized company names.
            rate_limiter_resources : dict[str, any]
                Request wait limiter to stop flooding.

            Returns
            -------
            tuple[cleaned text, both company names are present] : tuple[str, bool]
                The tuple of the cleaned text and validation status.
        """
        # Open the url and acquire the document content.
        # Error = Fatal, force exit from load function.
        raw_text = Processor.load_file_from_url(url, rate_limiter_resources);

        # Clean and truncate text
        cleaned_text = Processor.preprocess_text(raw_text);
        cleaned_text = Processor.normalize_text(cleaned_text);
        cleaned_text = cleaned_text[:450000]; # Reduce data load

        # Truncate to header to validate that we have the correct document
        lower_text = cleaned_text.lower()[:11000];

        # Check if both company names are present in the document headers
        found_companies = [name for name in company_names if re.search(r'\b' + re.escape(name) + r'\b', lower_text)];
        
        # Return the cleanedText if both company names are found, else False
        return cleaned_text, len(found_companies) == len(company_names);

    @staticmethod
    def getDocuments(
        source_links: list[str], 
        company_names: list[str],
        max_num_of_threads: int,
        rate_limiter_resources: dict[str, any] 
    ) -> list[Document]:
        """
            Acquire the document's text content from the source urls
            and validate the existence of both companies.

            Parameters
            ----------
            source_links : list[str]
                The list of documents source url.
            company_names : list[str]
                The list of company names.
            max_num_of_threads : int
                The defined maximum number of threads per thread pool.
            rate_limiter_resources : dict[str, any]
                Request wait limiter to stop flooding.

            Returns
            -------
            list : Document
                The list of document objects that contains the url and processed text content.
        """
        # Acquire company name's first word
        company_names_cut = [Processor.extract_all_but_last_word(name).lower() for name in company_names];

        # Create multiple threads to open & verify document
        with ThreadPoolExecutor(max_workers=max_num_of_threads) as thread_pool:
            futures = {
                thread_pool.submit(Processor.check_companies_in_document, url, company_names_cut, rate_limiter_resources): url
                for url in source_links
            };

            # Wait to fix race conditions causing no results for ones that should have results
            wait([future for future in futures], return_when=ALL_COMPLETED);

            # Wait for thread to finish processing and create new Document object
            documents = [];
            for future in as_completed(futures):
                url = futures[future];
                try:
                    cleaned_text, both_found = future.result();
                    if both_found:
                        documents.append(Document(url, cleaned_text));
                except Exception as e:
                    Logger.logMessage(f"[-] Error retrieving document for URL {url}: {e}");

        return documents;