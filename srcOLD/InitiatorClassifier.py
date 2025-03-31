from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import csv
import sys

from ContextAssistant import ContextAssistant
from Logger import Logger

class InitiatorClassifier:
    def __init__(self, companyAList: list, companyBList: list, assistant: ContextAssistant, threadCount: int = 5):
        self.companyAList = companyAList;
        self.companyBList = companyBList;
        self.assistant = assistant;
        self.threadCount = threadCount;

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
                writer.writerow(["INDEX", "INITIATOR", "DATEOFINITIATION", "REASON", "KEYFIGURES"]);
            
            for mainIndex in tqdm(
                range(self.__startIndex, self.__endIndex),
                desc = "\033[36mReading\033[0m",
                unit="items",
                ncols=80,
                bar_format="\033[92m{desc}: {percentage:3.0f}%|\033[92m{bar}\033[0m| {n_fmt}/{total_fmt} [elapsed: {elapsed}]\n"
            ):
                print("Reading index: ", mainIndex, "; Companies: ", self.companyAList[mainIndex], " & ", self.companyBList[mainIndex]);

                # Construct document file name & construct the folder constraint
                companyNames = [self.companyAList[mainIndex], self.companyBList[mainIndex]];
                formatDocName = f"{mainIndex}_{companyNames[0].replace(' ', '_')}_&_{companyNames[1].replace(' ', '_')}";

                batchStart = (mainIndex // 100) * 100;
                batchEnd = batchStart + 99;
                
                # Check if the file exists
                filePath = f"./DataSet/{batchStart}-{batchEnd}/{formatDocName}.txt";
                if not os.path.isfile(filePath):
                    print("Skipping: Document does not exist...")
                    continue;

                # Analyze the text
                try:
                    result = self.assistant.analyzeDocument(filePath);
                    writer.writerow([mainIndex, result["initiator"], result["date_of_initiation"], result["stated_reasons"], result["key_figures"]]);
                except Exception as e:
                    Logger.logMessage(f"[{Logger.get_current_timestamp()}] [-] Error analyzing text for index {mainIndex}: {e}");
                
                self.assistant.clearVectorStores();
