import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait, ALL_COMPLETED, CancelledError
from multiprocessing import Manager
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
from ChunkProcessor import ChunkProcessor

TEMP_DIRECTORY = "merge_extractor_temp";

"""
    - This object handles processing the document by:
        - Cleaning the data
        - Verifying the existence of both companies are present
        - Extracting the background section
"""
class Processor:
    def __init__(
            self, 
            assistant: BackupAssistant, 
            nlp: Language, 
            start_phrases: list[str], 
            thread_pool: ThreadPoolExecutor
        ):

        self.assistant = assistant;
        self.nlp = nlp;
        self.start_phrases = start_phrases;
        self.thread_pool = thread_pool;

    def __extract_all_but_last_word(self, company_name: str) -> str:
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

    def __load_file_from_url(self, url: str) -> str:
        """
            Acquire the document's text from the given url.

            Parameters
            ----------
            url : str
                The document source url.

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

        response = requests.get(url, headers=headers);
        if (response.text):
            return response.text;
        else:
            print(f"FATAL: Failed to load document via url. Err_Code: {response.status_code}");
            sys.exit(response.status_code);

    def __preprocess_text(self, content: str) -> str:
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

    def __normalize_text(self, text: str) -> str:
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

    def __check_companies_in_document(self, url: str, company_names: list[str]) -> tuple[str, bool]:
        """
            Validate the presence of both companys' name in the document.

            Parameters
            ----------
            url : str
                The document source url.
            company_names : list[str]
                The list of sanitized company names.

            Returns
            -------
            tuple[cleaned text, both company names are present] : tuple[str, bool]
                The tuple of the cleaned text and validation status.
        """
        # Open the url and acquire the document content.
        # Error = Fatal, force exit from load function.
        raw_text = self.__load_file_from_url(url);

        # Clean and truncate text
        cleaned_text = self.__preprocess_text(raw_text);
        cleaned_text = self.__normalize_text(cleaned_text);
        cleaned_text = cleaned_text[:450000]; # Reduce data load

        # Truncate to header to validate that we have the correct document
        lower_text = cleaned_text.lower()[:11000];

        # Check if both company names are present in the document headers
        found_companies = [name for name in company_names if re.search(r'\b' + re.escape(name) + r'\b', lower_text)];
        
        # Return the cleanedText if both company names are found, else False
        return cleaned_text, len(found_companies) == len(company_names);

    def getDocuments(self, source_links: list[str], company_names: list[str]) -> list[Document]:
        """
            Acquire the document's text content from the source urls
            and validate the existence of both companies.

            Parameters
            ----------
            source_links : list[str]
                The list of documents source url.
            company_names : list[str]
                The list of company names.

            Returns
            -------
            list : Document
                The list of document objects that contains the url and processed text content.
        """
        # Acquire company name's first word
        company_names_cut = [self.__extract_all_but_last_word(name).lower() for name in company_names];

        # Create multiple threads to open & verify document
        futures = {
            self.thread_pool.submit(self.__check_companies_in_document, url, company_names_cut):
            url for url in source_links
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
                Logger.logMessage(f"[{Logger.get_current_timestamp()}] [-] Error retrieving document for URL {url}: {e}");

        return documents;

    # Helper function for fallback method
    # def __analyzeDocumentWithObj(self, file_path: str, doc: Document):
    #     result = self.assistant.analyzeDocument(file_path);
    #     return result, doc;

    # def __processFallbackFutures(self, futures: list[tuple[str, Document]]) -> (Document | None):
    #     sectionFoundEvent = Event(); # Localized event to ttrack section discovery

    #     for result, doc in futures:
    #         if sectionFoundEvent.is_set():
    #             break;
        
    #         try:
    #             if result is None:
    #                 continue;

    #             match = re.search(r"\[(.*?)\]", result);
    #             foundSection = match.group(1) if match else "unknown";

    #             if foundSection == "Found":
    #                 sectionFoundEvent.set();  # Signal that a section has been found
    #                 return doc; # Returns the correct document with the detected section
    #         except Exception as e:
    #             Logger.logMessage(f"[{Logger.get_current_timestamp()}] [-] Error processing fallback future: {e}");

    #     return None;

    @staticmethod
    def process_document(
        doc: Document, 
        company_names: list[str], 
        main_index: int, 
        start_phrases: list, 
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
            start_phrases: list[str]
                The list of title variations for 'Background of the Merger' section.
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
            _, approx_chunks = ChunkProcessor.locateBackgroundChunk(doc.getContent(), start_phrases, nlp_model);
            if len(approx_chunks) > 0:
                if found_data.value: # Prevent race condition
                    return None;
            
                with lock:
                    if not found_data.value:
                        found_data.value = True;

                        # Write the found document with the 'Background' section into a new file
                        format_doc_name = f"{main_index}_{company_names[0].replace(' ', '_')}_&_{company_names[1].replace(' ', '_')}";
                        batch_start = (main_index // 100) * 100;
                        batch_end = batch_start + 99;
                        with open(f"./DataSet/{batch_start}-{batch_end}/{format_doc_name}.txt", "w", encoding="utf-8") as file:
                            file.write(f"URL: {doc.getUrl()}\n\n");
                            file.write(doc.getContent());

                        Logger.logMessage(f"[{Logger.get_current_timestamp()}] [+] Successfully created document for: {company_names[0]} & {company_names[1]}");
                        return doc.getUrl();
        except Exception as e:
            Logger.logMessage(f"[{Logger.get_current_timestamp()}] [-] Error processing {doc.getUrl()}: {e}");
        
        return None;

    def locateDocument(self, documents: list[Document], company_names: list[str], main_index: int) -> str | None:
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
        start_phrases = self.start_phrases; # extracted for multi-process safety
        
        # Extract nlp model name as multi-processing requires a seperate instaniation
        nlp_model = self.nlp.meta["lang"] + "_" + self.nlp.meta["name"];

        # Handle processing with and without multi-processing
        with Manager() as manager:
            found_data = manager.Value("b", False); # Shared boolean to only allow 1 final result
            lock = manager.Lock();

            # If only one document, process it directly without multiprocessing
            if len(documents) == 1:
                try:
                    return Processor.process_document(
                        documents[0], company_names, main_index, start_phrases, found_data, nlp_model, lock
                    );
                except Exception as e:
                    Logger.logMessage(f"[{Logger.get_current_timestamp()}] [-] Error processing {documents[0].getUrl()}: {e}");
                    return None;

            # Locate valid document with the 'Background of the Merger' section via multi-processing
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        Processor.process_document, doc, company_names, main_index, start_phrases, found_data, nlp_model, lock
                    ): doc
                    for doc in documents
                };

                # Catches the process_document results on the fly and validate result
                for future in as_completed(futures):
                    try:
                        doc_url = future.result();
                        if doc_url:
                            # Cancel remaining processes
                            for f in futures:
                                f.cancel();
                            return doc_url;
                    except Exception as e:
                        if isinstance(e, CancelledError):
                            continue;
                        doc = futures[future];
                        Logger.logMessage(f"[{Logger.get_current_timestamp()}] [-] Error processing {doc.getUrl()}: {e}");

        return None;

        # Fallback method if fuzzy fails. We will use openai to determine if the background section is within any of the documents
        # if not foundData:
        #     Logger.logMessage(f"[{Logger.get_current_timestamp()}] [*] No background section found for index {mainIndex}: {companyNames[0]} & {companyNames[1]}. Retrying via fallback...");

        #     # Create temp directory for creating temp file acceptable by openai
        #     os.makedirs(TEMP_DIRECTORY, exist_ok=True);
        #     fileDocComposite = [];

        #     # Create a list of async processes to force correct using openai
        #     for doc in documents:
        #         filePath = os.path.join(TEMP_DIRECTORY, f"merge_extractor_temp_{random.randint(1000, 99999)}.txt");
        #         with open(filePath, "w", encoding="utf-8") as file:
        #             file.write(doc.getContent());
        #             file.flush();
                
        #         # Wait for the file to be created; prevent race condition
        #         while not os.path.exists(filePath):
        #             time.sleep(0.1);
                
        #         fileDocComposite.append((filePath, doc));
            
        #     # Parallelly process all documents to locate "Background of the Merger" section via openai
        #     fallbackFutures = list(self.executor.map(lambda args: self.__analyzeDocumentWithObj(*args), fileDocComposite));

        #     fallbackResult = self.__processFallbackFutures(fallbackFutures);
        #     if fallbackResult is not None:
        #         # Write the data to a file
        #         batchStart = (mainIndex // 100) * 100;
        #         batchEnd = batchStart + 99;
        #         with open(f"./DataSet/{batchStart}-{batchEnd}/{formatDocName}.txt", "w", encoding="utf-8") as file:
        #             file.write(f"URL: {fallbackResult.getUrl()}\n\n");
        #             file.write(fallbackResult.getContent());
                
        #         if os.path.exists(TEMP_DIRECTORY):
        #             time.sleep(0.5);
        #             shutil.rmtree(TEMP_DIRECTORY);

        #         Logger.logMessage(f"[{Logger.get_current_timestamp()}] [+] Successfully created document for: {companyNames[0]} & {companyNames[1]}");
        #         return fallbackResult.getUrl();
            
        #     if os.path.exists(TEMP_DIRECTORY):
        #         time.sleep(0.5);
        #         shutil.rmtree(TEMP_DIRECTORY);
        #     return None;

        # return None; # Extreme backup