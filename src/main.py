# Main imports
import pandas as pd
from dotenv import load_dotenv
import os
import spacy
import torch

from Assistant import Assistant
from Crawler import Crawler

load_dotenv();

# Global variables
api_key = os.getenv("OPENAI_API_KEY");
csvFile = "./truncatedData.csv";
maxNumOfThreads = os.cpu_count();
deleteAssistant = False;

# Load spaCy model
if torch.cuda.is_available():
    spacy.prefer_gpu();

nlp = spacy.load("en_core_web_sm");

# Read the CSV file and extract the date & both merging companies (index base)
filedDate = pd.read_csv(csvFile, header=None).iloc[:, 1].tolist();
companyAList = pd.read_csv(csvFile, header=None).iloc[:, 2].tolist();
companyBList = pd.read_csv(csvFile, header=None).iloc[:, 3].tolist();

# Phrases for locating start point of the background section
startPhrases = [
    "Background of the transaction",
    "Background of the merger",
    "Background of the offer",
    "Background of the acquisition",
    "Background of the Offer and the Merger"
];

def main():
    prompt = (
        "Locate the 'Background of the Merger' "
        "(which could be phrased differently, such as 'Background of the Transaction', 'Background of the Acquisition', or 'Background of the Offer', and so on), "
        "which is the chronological timeline of events leading to the two companies' merger. "
        "With that section, determine who initiated the merger/deal first. "
        "Provide only the initiator's company name in the exact format with brackets so I can seperate them later: [company name] "
    );

    assistant = Assistant(api_key, prompt, "gpt-4o-mini");

    crawler = Crawler(filedDate, companyAList, companyBList, startPhrases, maxNumOfThreads, nlp, assistant);
    crawler.runCrawler(startIndex=0, endIndex=5);

    if deleteAssistant:
        assistant.deleteAssistant();


if __name__ == "__main__":
    main();