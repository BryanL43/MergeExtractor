from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from multiprocessing import get_context
import time
import gc
import os
import traceback
from tqdm import tqdm
import csv

from src.crawler.CrawlerSupport import CrawlerSupport
from src.utils.Logger import Logger
from src.utils.init_worker import init_worker

from src.dependencies.config import ANNOUNCEMENT_DATES, COMPANY_A_LIST, COMPANY_B_LIST

class CrawlerHandler:
    def __init__(self):
        pass;
    
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
                        [
                            main_index, 
                            ANNOUNCEMENT_DATES[main_index], 
                            COMPANY_A_LIST[main_index], 
                            COMPANY_B_LIST[main_index], 
                            url
                        ]
                    );
                except Exception as e:
                    Logger.logMessage(f"[-] Error writing to output for index {main_index}: {e}");

    def runCrawler(
        self, 
        start_index: int = None, 
        end_index: int = None, 
        index: int = None, 
        date_margin: int = 4, # Default; can be overwritten
        batch_size: int = None,
        max_calls_per_sec: int = 5 # SEC EDGAR only allows 10 requests per second
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
                COMPANY_A_LIST[i],
                COMPANY_B_LIST[i],
                ANNOUNCEMENT_DATES[i]
            )
            for i in range(self.__start_index, self.__end_index)
        ];

        total_tasks = len(jobs);
        
        # Process jobs in batches of 5
        acquired_documents = [];
        with tqdm(
            total=total_tasks,
            desc="\033[35mScraping\033[0m",
            unit="items",
            ncols=80,
            bar_format="\033[92m{desc}: {percentage:3.0f}%|\033[92m{bar}\033[0m| {n_fmt}/{total_fmt} [elapsed: {elapsed}]\n"
        ) as pbar:
            for i in range(0, total_tasks, self.__batch_size):
                batch_jobs = jobs[i:i + self.__batch_size];

                # Check if batch has only one job, then don't use multiprocessing
                if len(batch_jobs) == 1:
                    # Process the single job without multiprocessing
                    try:
                        result = CrawlerSupport.process_single_job(
                            batch_jobs[0],
                            date_margin
                        );
                        if result is not None:
                            main_index, doc_url = result;
                            acquired_documents.append((main_index, doc_url));
                            
                        pbar.update(1);
                    except Exception as e:
                        print(f"Error processing single job: {e}");
                        Logger.logMessage(f"[-] Process failed with error: {traceback.format_exc()}");
                else:
                    # Launches process pool for the current batch
                    with ProcessPoolExecutor(
                        max_workers=min(len(batch_jobs), os.cpu_count()),
                        initializer=init_worker,
                        initargs=(max_calls_per_sec,),
                        mp_context=get_context("spawn")
                    ) as process_pool:
                        futures = {
                            process_pool.submit(
                                CrawlerSupport.process_single_job, 
                                job, 
                                date_margin
                            ): job
                            for job in batch_jobs
                        };

                        # Track progress by waiting for task completion
                        for future in as_completed(futures):
                            try:
                                result = future.result();
                                if result is not None:
                                    main_index, doc_url = result;
                                    acquired_documents.append((main_index, doc_url));
                                
                                pbar.update(1);
                            except Exception as e:
                                print(f"Error processing batch job: {e}");
                                Logger.logMessage(f"[-] Process failed with error: {traceback.format_exc()}");

                if len(batch_jobs) > 1:
                    # Cooldown and resource flush after every batch
                    print(f"Completed batch {i // self.__batch_size + 1}, waiting for cooldown...");
                    gc.collect(); # CPU flush
                    time.sleep(2);
                    print("Cooldown complete, proceeding to next batch...");
        
        acquired_documents.sort(key=lambda x: x[0]);
        self.__write_output_urls(acquired_documents);
