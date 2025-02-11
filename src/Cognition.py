from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os

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

        print("Successfully initialized Cogition");

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
        
        

        for mainIndex in tqdm(
            range(self.__startIndex, self.__endIndex),
            desc="\033[35mReading\033[0m",
            unit="items",
            ncols=80,
            bar_format="\033[92m{desc}: {percentage:3.0f}%|\033[92m{bar}\033[0m| {n_fmt}/{total_fmt} [elapsed: {elapsed}]\n"
        ):
            batchStart = (mainIndex // 100) * 100;
            batchEnd = batchStart + 99;

            if os.path.isfile(f"./DataSet/{batchStart}-{batchEnd}")

            print("Reading index: ", mainIndex, "; Companies: ", self.companyAList[mainIndex], " & ", self.companyBList[mainIndex]);



