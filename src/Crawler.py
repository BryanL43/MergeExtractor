from datetime import datetime
import re
import requests
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from multiprocessing import Manager, get_context
from rapidfuzz import fuzz
import csv
import time
import gc
from spacy.language import Language
import os
import traceback
from tqdm import tqdm

from CrawlerSupport import CrawlerSupport
from Logger import Logger
from BackupAssistant import BackupAssistant
# from Processor import Processor
from RateLimiter import RateLimiter

class Crawler:
    def __init__(
        self,
        announcement_date: list[str],
        company_A_list: list[str],
        company_B_list: list[str],
        start_phrases: list[str],
        nlp_model: str,
        max_num_of_threads: int,
        # assistant: BackupAssistant,
        # rate_limiter: RateLimiter
    ):
        self.announcement_date = announcement_date;
        self.company_A_list = company_A_list;
        self.company_B_list = company_B_list;
        self.start_phrases = start_phrases;
        self.nlp_model = nlp_model;
        self.max_num_of_threads = max_num_of_threads;
        # self.assistant = assistant;
        # self.rate_limiter = rate_limiter;

        self.__form_types = ["PREM14A", "S-4", "SC 14D9", "SC TO-T"];

        # Instantiate the Processor to clean & analzye documents
        # self._processor = Processor(self.assistant, self.nlp, self.start_phrases, self.thread_pool, self.rate_limiter);

    def runCrawler(
        self, 
        start_index: int = None, 
        end_index: int = None, 
        index: int = None, 
        date_margin: int = 4, # Default; can be override
        batch_size: int = None,
        max_calls_per_sec: int = 9 # SEC EDGAR only allows 10 requests per second
    ):
        # Ensure valid batch_size parameter
        if batch_size is None and index is None:
            raise ValueError("batch_size must be defined.");

        # If index is provided, override startIndex and endIndex
        if index is not None:
            self.__start_index = index;
            self.__end_index = index + 1;
            self.__batch_size = 1; # Index provided means only 1 batch size is necessary
        else:
            # Ensure start_index and end_index are set properly
            if start_index is None or end_index is None:
                raise ValueError("start_index and end_index must be provided if index is not provided.");

            self.__start_index = start_index;
            self.__end_index = end_index + 1;
            self.__batch_size = batch_size;
        
        # Create a flat list of jobs to transport data
        jobs = [
            (
                i,
                self.company_A_list[i],
                self.company_B_list[i],
                self.announcement_date[i]
            )
            for i in range(self.__start_index, self.__end_index)
        ];

        # Create a manager for shared resources
        manager = multiprocessing.Manager();
        
        # Create rate limiter resources
        rate_limiter_resources = RateLimiter.create_resources(manager, max_calls_per_sec);
        
        # TO DO: HANDLE 1 BATCH SIZE; NO MULTI-PROCESS **********************

        # Launches process pool to process the contents of a batch concurrently
        with ProcessPoolExecutor(mp_context=get_context("spawn"), max_workers=min(self.__batch_size, os.cpu_count())) as process_pool:
            futures = {
                process_pool.submit(
                    CrawlerSupport.process_single_job, 
                    job, 
                    date_margin, 
                    self.__form_types,
                    self.max_num_of_threads,
                    # self.assistant, 
                    self.nlp_model,
                    rate_limiter_resources
                ): job
                for job in jobs
            };

            total_tasks = len(jobs);
            with tqdm(
                total=total_tasks,
                desc="\033[35mProcessing\033[0m",
                unit="items",
                ncols=80,
                bar_format="\033[92m{desc}: {percentage:3.0f}%|\033[92m{bar}\033[0m| {n_fmt}/{total_fmt} [elapsed: {elapsed}]\n"
            ) as pbar:
                # Track progress by waiting for task completion
                for future in as_completed(futures):
                    try:
                        future.result();
                        pbar.update(1);
                    except Exception as e:
                        print(f"Error located at runCrawler process pool: {e}");
                        Logger.logMessage(f"[-] Process failed with error: {traceback.format_exc()}");
        
            # # After submitting all processes
            # for future in as_completed(futures):
            #     try:
            #         future.result();
            #         elapsed_time = time.time() - start_time
            #         logging.info(f"jobs completed. Elapsed Time: {elapsed_time:.2f} seconds.");
            #     except Exception as e:
            #         Logger.logMessage(f"[-] Process failed with error: {traceback.print_exc()}");

        # Flatten the list of results into one list of documents
        # all_documents = [doc for result in results for doc in result];
        # print(all_documents);