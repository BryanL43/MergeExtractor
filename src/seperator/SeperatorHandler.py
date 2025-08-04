from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import get_context
from tqdm import tqdm
import os
from openai import OpenAI
from sentence_transformers import CrossEncoder
import traceback
import gc
import time

from src.utils.Logger import Logger
from src.dependencies.ChunkProcessor import ChunkProcessor
from src.dependencies.DatabaseHandler import DatabaseHandler

from src.dependencies.config import (
    OPENAI_API_KEY,
    RERANKER_MODEL,
    START_PHRASES,
    COMPANY_A_LIST,
    COMPANY_B_LIST,
    MAX_NUM_OF_THREADS
)

class SeperatorHandler:
    def __init__(self):
        pass;
    
    @staticmethod
    def process_single_doc(
        main_index: int,
        company_A: str,
        company_B: str
    ):
        print("Seperating passages for document index: ", main_index, "; Companies: ", company_A, " & ", company_B);

        # Instantiate utility objects in child process
        client = OpenAI(api_key=OPENAI_API_KEY);
        reranker = CrossEncoder(RERANKER_MODEL);
        chunk_processor = ChunkProcessor(reranker, client);

        company_names = [company_A, company_B];

        # Determine batch collection
        batch_start = (main_index // 100) * 100;
        batch_end = batch_start + 99;
        collection_name = f"batch_{batch_start}_{batch_end}";

        with DatabaseHandler() as db:
            dataset_collection = db.dataset_db[collection_name];
            extracted_collection = db.extracted_sections_db[collection_name];

            # Check if the original document exists
            doc = dataset_collection.find_one({"main_index": main_index});
            if not doc:
                print(f"Skipping index {main_index}: Document does not exist...");
                return;

            # Check if the extracted section already exists
            if extracted_collection.find_one({"main_index": main_index}):
                print(f"Skipping index {main_index}: Already processed and extracted...");
                return;

            # Use the stored unprocessed content
            text = doc["content"];

        with ThreadPoolExecutor(max_workers=MAX_NUM_OF_THREADS) as executor:
            try:
                chunks, approx_chunks = chunk_processor.locateBackgroundChunk(text, START_PHRASES, executor);
                if len(approx_chunks) == 0:
                    Logger.logMessage(f"[x] FLAGGED for Manual Inspection: Failed to locate a background chunk for index: {main_index}; Companies: {company_A} & {company_B}");
                    return;

                # Acquire ranked chunks according the the "Background" section
                section_passage = chunk_processor.getSectionPassage(chunks, approx_chunks, company_names, executor);
                if section_passage is None:
                    Logger.logMessage(f"[x] FLAGGED for Manual Inspection: Failed to acquire a section passage for index: {main_index}; Companies: {company_A} & {company_B}");
                    return;

                # Write the section passage
                with DatabaseHandler() as db:
                    collection = db.extracted_sections_db[collection_name];
                    passage = {
                        "main_index": main_index,
                        "company_A": company_A,
                        "company_B": company_B,
                        "content": section_passage
                    };
                    collection.insert_one(passage);

            except Exception as e:
                Logger.logMessage(f"[-] Error: {e}");
                Logger.logMessage(traceback.format_exc(), time_stamp=False);

    def runSeperator(
        self,
        start_index: int = None,
        end_index: int = None,
        index: int = None,
        batch_size: int = None # Range [1-3 is recommended] to not flood the API
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
            # Ensure startIndex and endIndex are set properly
            if start_index is None or end_index is None:
                raise ValueError("start_index and end_index must be provided if index is not provided.");

            self.__start_index = start_index;
            self.__end_index = end_index + 1;
            self.__batch_size = batch_size;

        indices_to_process = list(range(self.__start_index, self.__end_index));
        total_tasks = len(indices_to_process);

        with tqdm(
            total=total_tasks,
            desc = "\033[36mSeperating\033[0m",
            unit="items",
            ncols=80,
            bar_format="\033[92m{desc}: {percentage:3.0f}%|\033[92m{bar}\033[0m| {n_fmt}/{total_fmt} [elapsed: {elapsed}]\n"
        ) as pbar:
            # Process jobs in batches of batch_size
            for i in range(0, total_tasks, self.__batch_size):
                batch_jobs = indices_to_process[i:i + self.__batch_size];

                # Check if the batch has only one job, then don't use multiprocessing
                if len(batch_jobs) == 1:
                    # Process the single job without multiprocessing
                    try:
                        SeperatorHandler.process_single_doc(
                            batch_jobs[0],
                            COMPANY_A_LIST[batch_jobs[0]],
                            COMPANY_B_LIST[batch_jobs[0]]
                        );

                        pbar.update(1);
                    except Exception as e:
                        print(f"Error processing single job: {e}");
                        Logger.logMessage(f"[-] Process failed with error: {traceback.format_exc()}");
                else:
                    # Launches process pool for the current batch
                    with ProcessPoolExecutor(mp_context=get_context("spawn"), max_workers=min(self.__batch_size, os.cpu_count())) as process_pool:
                        futures = {
                            process_pool.submit(
                                SeperatorHandler.process_single_doc,
                                job,
                                COMPANY_A_LIST[job],
                                COMPANY_B_LIST[job]
                            ): job
                            for job in batch_jobs
                        };

                        # Catch exceptions but nothing is returned
                        for future in as_completed(futures):
                            try:
                                future.result();
                                pbar.update(1);
                            except Exception as e:
                                print(f"Error processing future: {e}");
                                Logger.logMessage(f"[-] Process failed with error: {traceback.format_exc()}");

                # Cooldown and resource flush after every batch
                if len(batch_jobs) > 1:
                    print(f"Completed batch {i // self.__batch_size + 1}, waiting for cooldown...");
                    gc.collect();  # CPU flush
                    time.sleep(2);
                    print("Cooldown complete, proceeding to next batch...");

