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
    print("spaCy is using GPU");
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
    # prompt = (
    #     "Locate the 'Background of the Merger' (which could be phrased differently, such as "
    #     "'Background of the Transaction', 'Background of the Acquisition', or 'Background of the Offer', and so on), "
    #     "which is the chronological timeline of events leading to the two companies' merger. "
    #     "With that section, determine who was the first entity to **actively initiate or facilitate** discussions leading to the merger/deal. "
    #     "This could be one of the companies involved or a third party, such as an investor, advisor, consultant, or financial institution. "
    #     "If an external entity introduced the two companies or arranged the initial discussions, return their name instead. "
    #     "Provide only the initiator's name in the exact format with brackets so I can separate them later: [initiator name]."
    # );

    # prompt = (
    #     "Locate the 'Background of the Merger' section (which may be titled differently, such as "
    #     "'Background of the Transaction', 'Background of the Acquisition', or 'Background of the Offer'). "
    #     "This section provides a chronological timeline of events leading to the merger. "
    #     "Identify the first entity that **actively initiated or facilitated** discussions leading to the merger or acquisition. "
    #     "The initiator must be a company or financial institution, **not an individual person**. "
    #     "Read the beginning of the document if the initiator is an individual or an entity that is not a company. "
    #     "If a third-party advisor, consultant, or financial institution arranged the initial discussions, return their name instead. "
    #     "Strictly return only the initiating company's name within brackets: [initiator name]. "
    # );

    prompt = (
        "Locate the 'Background of the Merger' section (which may be titled differently, such as "
        "'Background of the Transaction', 'Background of the Acquisition', or 'Background of the Offer'). "
        "This section provides a chronological timeline of events leading to the merger. "
        "If you cannot find the section, strictly return None within brackets like this: [None]. "
        "Identify the first entity that **actively initiated or facilitated** discussions leading to the merger or acquisition. "
        "The initiator must be a company or financial institution, **not an individual person**. "
        "If the initial mention is an abbreviation or placeholder (e.g., 'Parent'), read the beginning of the document "
        "to locate the corresponding full company name (for example, 'Ocean Energy, Inc.') and use that instead. "
        "If a third-party advisor, consultant, or financial institution arranged the initial discussions, return their name instead. "
        "Strictly return only the initiating company's name within brackets: [initiator name]. "
        "Chain-of-thought: "
        "Step 1: Locate the 'Background' section. "
        "Step 2: Identify the first mention of an initiating entity. "
        "Step 3: Check if the entity is provided as an abbreviation (e.g., 'Parent'). "
        "Step 4: If it is, scan the beginning of the document for the full company name that maps to this abbreviation. "
        "Step 5: Return the full name within brackets."
    );

    assistant = Assistant(api_key, prompt, "gpt-4o-mini");

    crawler = Crawler(filedDate, companyAList, companyBList, startPhrases, maxNumOfThreads, nlp, assistant);
    # crawler.runCrawler(startIndex=0, endIndex=10);
    crawler.runCrawler(index=6);

    if deleteAssistant:
        assistant.deleteAssistant();


if __name__ == "__main__":
    main();