from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import csv
import re
import sys
from sentence_transformers import SentenceTransformer
import nltk

# Install the required NLTK data
for token in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{token}");
    except LookupError:
        nltk.download(token);

from nltk.tokenize import sent_tokenize

from Assistant import Assistant
from Logger import Logger

class Cognition:
    def __init__(
            self,
            companyAList: list,
            companyBList: list,
            threadCount: int,
            assistant: Assistant
        ):

        self.companyAList = companyAList;
        self.companyBList = companyBList;
        self.threadCount = threadCount;
        self.assistant = assistant;

        self.executor = ThreadPoolExecutor(max_workers=self.threadCount);

        print("Successfully initialized Cognition");

    def findInitiator(self, startIndex: int = None, endIndex: int = None, index: int = None):
        # If index is provided, override startIndex and endIndex
        if index is not None:
            self.__startIndex = index;
            self.__endIndex = index + 1;
        else:
            # Ensure startIndex and endIndex are set properly
            if startIndex is None or endIndex is None:
                raise ValueError("startIndex and endIndex must be provided if index is not provided.");

            self.__startIndex = startIndex;
            self.__endIndex = endIndex + 1;

        fileExists = os.path.isfile("outputUnion.csv");
        with open("outputUnion.csv", mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file);
            if not fileExists:
                writer.writerow(["INDEX", "INITIATOR", "REASON"])
            
            for mainIndex in tqdm(
                range(self.__startIndex, self.__endIndex),
                desc = "\033[36mReading\033[0m",
                unit="items",
                ncols=80,
                bar_format="\033[92m{desc}: {percentage:3.0f}%|\033[92m{bar}\033[0m| {n_fmt}/{total_fmt} [elapsed: {elapsed}]\n"
            ):
                print("Reading index: ", mainIndex, "; Companies: ", self.companyAList[mainIndex], " & ", self.companyBList[mainIndex]);

                companyNames = [self.companyAList[mainIndex], self.companyBList[mainIndex]];
                formatDocName = f"{mainIndex}_{companyNames[0].replace(' ', '_')}_&_{companyNames[1].replace(' ', '_')}";

                batchStart = (mainIndex // 100) * 100;
                batchEnd = batchStart + 99;

                filePath = f"./DataSet/{batchStart}-{batchEnd}/{formatDocName}.txt";
                if not os.path.isfile(filePath):
                    continue;

                result = self.assistant.analyzeDocument(filePath);
                match = re.search(r"\[(.*?)\]", result);
                initiator = match.group(1) if match else "None";

                reason = result.replace(f"[{initiator}]", "").replace("\n", " ").strip() if match else "No reasoning provided";

                if initiator == "None":
                    continue;
                
                try:   
                    # Write to the output CSV
                    writer.writerow([f"index_{mainIndex}", initiator, reason]);
                except Exception as e:
                    Logger.logMessage(f"[{Logger.get_current_timestamp()}] [-] Error writing to outputUnion for index {mainIndex}: {e}");
                    self.assistant.clearVectorStores();

        # Clean up the vector store at the end as we can't clear while in parallel processing
        self.assistant.clearVectorStores();