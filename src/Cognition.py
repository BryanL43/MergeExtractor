from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import csv
import re
import sys
import torch
from sentence_transformers import SentenceTransformer
import nltk

# Install the required NLTK data
for token in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{token}");
    except LookupError:
        nltk.download(token);

from nltk.tokenize import sent_tokenize

from TextAnalyzerAssistant import TextAnalyzerAssistant
from Logger import Logger

device = "cuda" if torch.cuda.is_available() else "cpu";

class Cognition:
    def __init__(
            self,
            companyAList: list,
            companyBList: list,
            assistant: TextAnalyzerAssistant,
            threadCount: int = 5,
            chunkSize: int = 20,
            modelName: str = "all-MiniLM-L6-v2"
        ):

        self.companyAList = companyAList;
        self.companyBList = companyBList;
        self.assistant = assistant;
        self.threadCount = threadCount;
        self.chunkSize = chunkSize;
        self.modelName = modelName;

        self.executor = ThreadPoolExecutor(max_workers=self.threadCount);
        self.__model = SentenceTransformer(self.modelName, device=device);

        print("Successfully initialized Cognition");
    
    def __getEmbedding(self, text: str) -> torch.Tensor:
        """Embeds the given text as a tensor with SentenceTransformer"""
        return self.__model.encode(text, batch_size=8, convert_to_tensor=True).clone().detach().to(device);

    def __splitChunk(self, text, chunk_size=20):
        """Convert the text into smaller chunks"""
        sentences = sent_tokenize(text);
        return [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)];

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
                writer.writerow(["INDEX", "INITIATOR", "REASON"]);
            
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
                    continue;

                with open(filePath, "r", encoding="utf-8") as file:
                    content = file.read();

                # Split the content into smaller chunks & embed it
                chunks = self.__splitChunk(content, self.chunkSize);
                chunkEmbeddings = torch.stack([self.__getEmbedding(chunk) for chunk in chunks]);

                # Query embedding
                queryEmbedding = self.__getEmbedding(self.assistant.getQuery());

                # Calculate cosine similarity
                similarities = torch.cosine_similarity(queryEmbedding, chunkEmbeddings, dim=1);

                # Get top 3 most relevant chunks
                topIndices = torch.argsort(similarities, descending=True)[:3];
                topChunks = [chunks[i] for i in topIndices];

                content = "\n\n".join(topChunks) + "\n\nWho initiated the merger?";

                # Analyze the text
                result = self.assistant.analyzeText(content);
                print(result);

                
                # try:   
                #     # Write to the output CSV
                #     writer.writerow([f"index_{mainIndex}", initiator, reason]);
                # except Exception as e:
                #     Logger.logMessage(f"[{Logger.get_current_timestamp()}] [-] Error writing to outputUnion for index {mainIndex}: {e}");
                #     self.assistant.clearVectorStores();
