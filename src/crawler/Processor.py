import re
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED, CancelledError
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

from src.utils.Logger import Logger
from src.utils.Document import Document
from src.dependencies.ChunkProcessor import ChunkProcessor
from src.dependencies.RateLimiter import RateLimiter

from src.dependencies.config import START_PHRASES, MAX_NUM_OF_THREADS

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
        with ThreadPoolExecutor(max_workers=MAX_NUM_OF_THREADS) as thread_pool:
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

    @staticmethod
    def process_document(
        doc: Document, 
        company_names: list[str], 
        main_index: int, 
        found_data: bool, 
        nlp_model: str, 
        lock: any
    ) -> str | None:
        """
            Process a single document to locate 'Background of the Merger' section and write it to a file.

            Parameters
            ----------
            doc : Document
                The document object containing the url and its content.
            company_names : list[str]
                The list of company names.
            main_index : int
                The current index of the data we are processing.
            found_data : bool
                State that tracks whether we have located a document with the 'Background of the Merger' section.
            nlp_model : str
                The nlp model's name that'll be instantiated seperately for multi-processing.
            lock : Lock
                The lock that only permits one final result.

            Returns
            -------
            str
                The document url that contains the 'Background of the Merger' section.
        """
        try:
            _, approx_chunks = ChunkProcessor.locateBackgroundChunk(doc.getContent(), [phrase for phrase in START_PHRASES if phrase != "Background"], nlp_model);
            if len(approx_chunks) == 0: # Fallback with lower confidence in accuracy
                _, approx_chunks = ChunkProcessor.locateBackgroundChunk(doc.getContent(), ["Background"], nlp_model);

            if len(approx_chunks) > 0:
                if found_data.value: # Prevent race condition
                    return None;
            
                with lock:
                    if not found_data.value:
                        found_data.value = True;
                        time.sleep(1);

                        # Write the found document with the 'Background' section into a new file
                        format_doc_name = f"{main_index}_{company_names[0].replace(' ', '_')}_&_{company_names[1].replace(' ', '_')}";
                        batch_start = (main_index // 100) * 100;
                        batch_end = batch_start + 99;
                        with open(f"./DataSet/{batch_start}-{batch_end}/{format_doc_name}.txt", "w", encoding="utf-8") as file:
                            file.write(f"URL: {doc.getUrl()}\n\n");
                            file.write(doc.getContent());

                        Logger.logMessage(f"[+] Successfully created document for: {company_names[0]} & {company_names[1]}");
                        return doc.getUrl();
        except (CancelledError, EOFError, FileNotFoundError):
            # Ignore these errors since they occur when processes are being stopped, results can be discarded
            return None;
        except Exception as e:
            Logger.logMessage(f"[-] Error processing {doc.getUrl()}: {e}");
        
        return None;

    # @staticmethod
    # def fallback_check(
    #     documents: list[Document], 
    #     company_names: list[str], 
    #     main_index: int, 
    #     max_num_of_threads: int, 
    #     assistant: BackupAssistant
    # ) -> (str | None):
    #     # Create temp directory for creating temp file acceptable by openai
    #     os.makedirs(TEMP_DIRECTORY, exist_ok=True);
    #     file_doc_pairs = [];

    #     # Create the temp files
    #     for doc in documents:
    #         file_path = os.path.join(TEMP_DIRECTORY, f"merge_extractor_temp_{random.randint(1000, 99999)}.txt");
    #         with open(file_path, "w", encoding="utf-8") as file:
    #             file.write(doc.getContent());
    #             file.flush();
            
    #         # Wait for the file to be created; prevent race condition
    #         while not os.path.exists(file_path):
    #             time.sleep(0.1);
            
    #         file_doc_pairs.append((file_path, doc));

    #     # Parallely process the documents via openai
    #     with ThreadPoolExecutor(max_workers=max_num_of_threads) as thread_pool:
    #         futures = [
    #             thread_pool.submit(
    #                 lambda pair: (assistant.analyzeDocument(pair[0]), pair[1]),  # Returns tuple of (result, doc)
    #                 pair
    #             ) for pair in file_doc_pairs
    #         ]

    #         section_found = Event();  # Tracks first section discovery
    #         fallback_result = None;

    #         # Process the futures as they complete; terminates the moment the first result is found
    #         for future in as_completed(futures):
    #             if section_found.is_set():
    #                 future.cancel();
    #                 continue;
            
    #             try:
    #                 result, doc = future.result();
    #                 if result is None:
    #                     continue;

    #                 # Process the result message
    #                 match = re.search(r"\[(.*?)\]", result);
    #                 foundSection = match.group(1) if match else "unknown";

    #                 if foundSection == "Found":
    #                     section_found.set();  # Signal that a section has been found
    #                     fallback_result = doc;
    #                     break;
    #             except Exception as e:
    #                 Logger.logMessage(f"[-] Error processing fallback future: {e}");

    #         # Terminate all remaining futures
    #         if section_found.is_set():
    #             for f in futures:
    #                 f.cancel();
            
    #         if fallback_result:
    #             # Write the data to a file
    #             format_doc_name = f"{main_index}_{company_names[0].replace(' ', '_')}_&_{company_names[1].replace(' ', '_')}";
    #             batchStart = (main_index // 100) * 100;
    #             batchEnd = batchStart + 99;
    #             with open(f"./DataSet/{batchStart}-{batchEnd}/{format_doc_name}.txt", "w", encoding="utf-8") as file:
    #                 file.write(f"URL: {fallback_result.getUrl()}\n\n");
    #                 file.write(fallback_result.getContent());
                
    #             # Delete the temp directory
    #             if os.path.exists(TEMP_DIRECTORY):
    #                 time.sleep(0.5);
    #                 shutil.rmtree(TEMP_DIRECTORY);

    #             Logger.logMessage(f"[+] Retry attempt successfully created document for: {company_names[0]} & {company_names[1]}");
    #             return fallback_result.getUrl();

    #     # Delete the temp directory
    #     if os.path.exists(TEMP_DIRECTORY):
    #         time.sleep(0.5);
    #         shutil.rmtree(TEMP_DIRECTORY);
        
    #     return None;

    @staticmethod
    def locateDocument(
        documents: list[Document], 
        company_names: list[str], 
        main_index: int
    ) -> str | None:
        """
            Acquires the document with the 'Background of the Merger' section and returns its url.

            Parameters
            ----------
            documents : list[Document]
                The list of document object containing the url and its content.
            company_names : list[str]
                The list of company names.
            main_index : int
                The current index of the data we are processing.

            Returns
            -------
            str
                The document url that contains the 'Background of the Merger' section.
        """
        # Handle processing with and without multi-processing
        with Manager() as manager:
            found_data = manager.Value("b", False); # Shared boolean to only allow 1 final result
            lock = manager.Lock();

            # If only one document, process it directly without multiprocessing
            if len(documents) == 1:
                try:
                    return Processor.process_document(
                        documents[0], company_names, main_index, found_data, nlp_model, lock, max_num_of_threads
                    );
                except Exception as e:
                    Logger.logMessage(f"[-] Error processing {documents[0].getUrl()}: {e}");
                    return None;

            # Locate valid document with the 'Background of the Merger' section via multi-processing
            with ThreadPoolExecutor(max_workers=MAX_NUM_OF_THREADS) as executor:
                futures = {
                    executor.submit(
                        Processor.process_document, doc, company_names, main_index, found_data, nlp_model, lock, max_num_of_threads
                    ): doc
                    for doc in documents
                };

                # Catches the process_document results on the fly and validate result
                for future in as_completed(futures):
                    try:
                        doc_url = future.result();
                        if doc_url:
                            # Cancel remaining processes and force termination
                            executor.shutdown(wait=False, cancel_futures=True);
                            return doc_url;
                    except Exception as e:
                        if isinstance(e, (CancelledError, EOFError)):
                            continue;
                        doc = futures[future];
                        Logger.logMessage(f"[-] Error processing {doc.getUrl()}: {e}");
        
        # Fallback method if fuzzy fails. We will use openai to determine if the background section is within any of the documents
        Logger.logMessage(
            f"[*] No background section found for index {main_index}: {company_names[0]} & {company_names[1]}. Retrying via fallback..."
        );

        # Instantiate backup assistant only when necessary
        # assistant = BackupAssistant(api_key, "Backup Assistant", "gpt-4o-mini");
        # fallback_result = Processor.fallback_check(documents, company_names, main_index, max_num_of_threads, assistant);
        # if fallback_result is None:
        #     return None;
    
        # return fallback_result;

        return None;