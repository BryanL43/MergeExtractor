from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import time
import traceback
import json
import os
import csv

from src.utils.Logger import Logger
from src.dependencies.DatabaseHandler import DatabaseHandler

from src.dependencies.config import (
    OPENAI_API_KEY,
    MAX_NUM_OF_THREADS,
    COMPANY_A_LIST,
    COMPANY_B_LIST,
    IDENTIFIER_MODEL,
    IDENTIFIER_TOOLS
)

class InitiatorIdentifier:
    def __init__(self):
        pass;

    def __write_result(self, acquired_results: list[tuple[int, dict]]):
        print("Writing results to CSV...");

        # Check if the file exists (to write headers if it's the first time)
        file_exists = os.path.isfile("outputUnion.csv");

        with open("outputUnion.csv", mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file);

            # Write the header row if the file doesn't exist
            if not file_exists:
                writer.writerow(["INDEX", "INITIATOR", "DATE_OF_INITIATION", "TYPE_OF_INITIATION", "REASON"]);

            for main_index, json_result in acquired_results:
                try:
                    # Dereference the json result for writing to csv
                    initiator = json_result.get("initiator", "");
                    date_of_initiation = json_result.get("date_of_initiation", "");
                    type_of_initiation = json_result.get("type_of_initiation", "");
                    reason = json_result.get("stated_reasons", "");

                    # Write the row for this result
                    writer.writerow([main_index, initiator, date_of_initiation, type_of_initiation, reason]);
                except Exception as e:
                    print(f"Error extracting or writing result for index {main_index}: {e}");

    def __process_document(self, main_index: int, client: OpenAI) -> tuple[int, dict] | None:
        try:
            # Construct the collection name
            batch_start = (main_index // 100) * 100;
            batch_end = batch_start + 99;
            collection_name = f"batch_{batch_start}_{batch_end}";

            with DatabaseHandler() as db:
                extracted_collection = db.extracted_sections_db[collection_name];

                # Check if the extracted section exists in MongoDB
                doc = extracted_collection.find_one({"main_index": main_index});
                if not doc:
                    print(f"Skipping index {main_index}: Extracted section document does not exist...");
                    return None;

                # Get the text from the MongoDB document
                text = doc["content"];

            # Acquire the initiator via querying the LLM
            response = client.chat.completions.create(
                model=IDENTIFIER_MODEL,
                messages=[{"role": "user", "content": text}],
                tools=IDENTIFIER_TOOLS,
                tool_choice="auto"
            );

            # Acquire the response from LLM
            tool_calls = response.choices[0].message.tool_calls;
            if tool_calls:
                args = json.loads(tool_calls[0].function.arguments);
                return (main_index, args);

        except Exception as e:
            Logger.logMessage(f"[-] Identifier failed to process index {main_index}: {traceback.format_exc()}");

        return None;

    def runIdentifier(
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

        client = OpenAI(api_key=OPENAI_API_KEY);

        acquired_results = [];
        with tqdm(
            total=total_tasks,
            desc = "\033[33mIdentifying Initiators\033[0m",
            unit="items",
            ncols=80,
            bar_format="\033[92m{desc}: {percentage:3.0f}%|\033[92m{bar}\033[0m| {n_fmt}/{total_fmt} [elapsed: {elapsed}]\n"
        ) as pbar, ThreadPoolExecutor(max_workers=MAX_NUM_OF_THREADS) as thread_pool:
            # Process jobs in batches of batch_size
            for i in range(0, total_tasks, self.__batch_size):
                batch_jobs = indices_to_process[i:i + self.__batch_size];

                # Check if the batch has only one job, then don't use multiprocessing
                if len(batch_jobs) == 1:
                    main_index = batch_jobs[0];

                    result = self.__process_document(main_index, client);
                    if result:
                        acquired_results.append(result);

                    pbar.update(1);
                else:
                    # Process the batch in parallel
                    futures = {
                        thread_pool.submit(self.__process_document, main_index, client): main_index
                        for main_index in batch_jobs
                    };

                    # Catch LLM responses on the fly
                    for future in as_completed(futures):
                        try:
                            result = future.result();
                            if result:
                                acquired_results.append(result);

                            pbar.update(1);
                        except Exception as e:
                            print(f"Error processing future: {e}");
                            Logger.logMessage(f"[-] Future error with identifying initiator: {traceback.format_exc()}");

                # Cooldown and resource flush after every batch
                if len(batch_jobs) > 1:
                    print(f"Completed batch {i // self.__batch_size + 1}, waiting for cooldown...");
                    time.sleep(2);
                    print("Cooldown complete, proceeding to next batch...");

        acquired_results.sort(key=lambda x: x[0]);
        self.__write_result(acquired_results);
