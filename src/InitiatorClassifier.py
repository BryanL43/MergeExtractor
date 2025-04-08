from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import csv
import sys
from spacy.language import Language
from openai import OpenAI
from sentence_transformers import CrossEncoder
import traceback

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
            thread_pool: ThreadPoolExecutor, 
            nlp: Language, 
            reranker_model: CrossEncoder,
            assistant: AnalysisAssistant
        ):
        self.client = OpenAI(api_key=api_key);
        self.company_A_list = company_A_list;
        self.company_B_list = company_B_list;
        self.start_phrases = start_phrases;
        self.thread_pool = thread_pool;
        self.nlp = nlp;
        self.reranker_model = reranker_model;
        self.assistant = assistant;
    
        # Instantiate the ChunkProcessor to locate relevant chunk
        self._chunk_processor = ChunkProcessor(self.nlp, self.reranker_model, self.client, self.thread_pool);
    
    def __write_result(self, main_index: int, result: dict):
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

    def findInitiator(self, start_index: int = None, end_index: int = None, index: int = None):
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

        for main_index in tqdm(
            range(self.__start_index, self.__end_index),
            desc = "\033[36mReading\033[0m",
            unit="items",
            ncols=80,
            bar_format="\033[92m{desc}: {percentage:3.0f}%|\033[92m{bar}\033[0m| {n_fmt}/{total_fmt} [elapsed: {elapsed}]\n"
        ):
            print("Reading index: ", main_index, "; Companies: ", self.company_A_list[main_index], " & ", self.company_B_list[main_index]);

            # Construct document file name & construct the folder constraint
            company_names = [self.company_A_list[main_index], self.company_B_list[main_index]];
            format_doc_name = f"{main_index}_{company_names[0].replace(' ', '_')}_&_{company_names[1].replace(' ', '_')}";

            batch_start = (main_index // 100) * 100;
            batch_end = batch_start + 99;
            
            # Check if the file exists
            file_path = f"./DataSet/{batch_start}-{batch_end}/{format_doc_name}.txt";
            if not os.path.isfile(file_path):
                print("Skipping: Document does not exist...");
                continue;
        
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read();

            try:
                chunks, approx_chunks = self._chunk_processor.locateBackgroundChunk(text, self.start_phrases);
                if len(approx_chunks) == 0:
                    print("FATAL: Failed to locate a background chunk for index: ", main_index, "; Companies: ", self.company_A_list[main_index], " & ", self.company_B_list[main_index]);
                    sys.exit(1);
                
                section_passage = self._chunk_processor.getSectionPassage(chunks, approx_chunks, self.start_phrases, company_names);
                if section_passage is None:
                    print("FATAL: Failed to acquire a section passage for index: ", main_index, "; Companies: ", self.company_A_list[main_index], " & ", self.company_B_list[main_index]);
                    sys.exit(1);
                
                # Write the section passage for debugging
                section_file_path = f"./ExtractedSection/{batch_start}-{batch_end}/{format_doc_name}.txt";
                with open(section_file_path, "w", encoding="utf-8") as file:
                    file.write(section_passage);
            
                # result = self.assistant.analyzeDocument(section_passage);
                # self.__write_result(main_index, result);
                
            except Exception as e:
                Logger.logMessage(f"[{Logger.get_current_timestamp()}] [-] Error: {e}");
                Logger.logMessage(traceback.format_exc())
                sys.exit(1);