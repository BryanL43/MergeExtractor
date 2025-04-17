from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from tqdm import tqdm
import os
import csv
import sys
from openai import OpenAI
from sentence_transformers import CrossEncoder
import traceback
import gc
import time

from AnalysisAssistant import AnalysisAssistant
from Logger import Logger
from ChunkProcessor import ChunkProcessor

class InitiatorClassifier:
    def __init__(
            self, 
            api_key: str,
            company_A_list: list[str], 
            company_B_list: list[str], 
            start_phrases: list[str],
            nlp_model: str, 
            max_num_of_threads: int, 
            reranker_model: str,
            assistant: AnalysisAssistant
        ):
        self.api_key = api_key
        self.company_A_list = company_A_list;
        self.company_B_list = company_B_list;
        self.start_phrases = start_phrases;
        self.max_num_of_threads = max_num_of_threads;
        self.nlp_model = nlp_model;
        self.reranker_model = reranker_model;
        self.assistant = assistant;
    
    @staticmethod
    def write_result(main_index: int, result: dict):
        file_exists = os.path.isfile("outputUnion.csv");
        with open("outputUnion.csv", mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file);
            if not file_exists:
                writer.writerow(["INDEX", "INITIATOR", "DATE_OF_INITIATION", "TYPE_OF_INITIATION", "REASON", "KEY_FIGURES"]);
            
            writer.writerow(
                [
                    main_index, 
                    result["initiator"], 
                    result["date_of_initiation"], 
                    result["type_of_initiation"], 
                    result["stated_reasons"], 
                    result["key_figures"]
                ]
            );
    
    @staticmethod
    def process_single_doc(
        main_index: int, 
        company_A: str, 
        company_B: str, 
        start_phrases: list[str],
        max_num_of_threads: int,
        nlp_model: str,
        reranker_model: str,
        api_key: str,
        assistant: AnalysisAssistant
    ):
        print("Reading index: ", main_index, "; Companies: ", company_A, " & ", company_B);

        # Instantiate utility objects in child process
        client = OpenAI(api_key=api_key);
        reranker = CrossEncoder(reranker_model);
        chunk_processor = ChunkProcessor(reranker, client);

        # Construct document file name & construct the folder constraint
        company_names = [company_A, company_B];
        format_doc_name = f"{main_index}_{company_names[0].replace(' ', '_')}_&_{company_names[1].replace(' ', '_')}";

        batch_start = (main_index // 100) * 100;
        batch_end = batch_start + 99;
        
        # Check if the file exists
        file_path = f"./DataSet/{batch_start}-{batch_end}/{format_doc_name}.txt";
        if not os.path.isfile(file_path):
            print(f"Skipping {main_index}: Document does not exist...");
            return;
        
        # Check if the extracted file exists
        extracted_path = f"./ExtractedSection/{batch_start}-{batch_end}/{format_doc_name}.txt";
        if os.path.isfile(extracted_path):
            print(f"Skipping {extracted_path}: Already processed and extracted...");
            return;
    
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read();

        try:
            chunks, approx_chunks = chunk_processor.locateBackgroundChunk(text, start_phrases, max_num_of_threads, nlp_model);
            if len(approx_chunks) == 0:
                print("FATAL: Failed to locate a background chunk for index: ", main_index, "; Companies: ", company_A, " & ", company_B);
                sys.exit(1);
            
            # Acquire ranked chunks according the the "Background" section
            section_passage = chunk_processor.getSectionPassage(chunks, approx_chunks, company_names, max_num_of_threads);
            if section_passage is None:
                print("FATAL: Failed to acquire a section passage for index: ", main_index, "; Companies: ", company_A, " & ", company_B);
                sys.exit(1);
            
            # Write the section passage for debugging
            with open(extracted_path, "w", encoding="utf-8") as file:
                file.write(section_passage);
        
            result = assistant.analyzeDocument(section_passage);
            InitiatorClassifier.write_result(main_index, result);
            
        except Exception as e:
            Logger.logMessage(f"[-] Error: {e}");
            Logger.logMessage(traceback.format_exc(), time_stamp=False);
            sys.exit(1);

    def findInitiator(
        self, 
        start_index: int = None, 
        end_index: int = None, 
        index: int = None, 
        batch_size: int = None,
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
            desc = "\033[36mReading\033[0m",
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
                        InitiatorClassifier.process_single_doc(
                            batch_jobs[0],
                            self.company_A_list[batch_jobs[0]],
                            self.company_B_list[batch_jobs[0]],
                            self.start_phrases, 
                            self.max_num_of_threads, 
                            self.nlp_model,
                            self.reranker_model,
                            self.api_key,
                            self.assistant
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
                                InitiatorClassifier.process_single_doc,
                                job,
                                self.company_A_list[job],
                                self.company_B_list[job],
                                self.start_phrases,
                                self.max_num_of_threads,
                                self.nlp_model,
                                self.reranker_model,
                                self.api_key,
                                self.assistant
                            ): job
                            for job in batch_jobs
                        };

                        # Track progress by waiting for task completion
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
                    gc.collect()  # CPU flush
                    time.sleep(2);
                    print("Cooldown complete, proceeding to next batch...");
