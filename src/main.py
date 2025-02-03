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
    # prompt = (
    #     "Locate the 'Background of the Merger' (which could be phrased differently, such as "
    #     "'Background of the Transaction', 'Background of the Acquisition', or 'Background of the Offer', and so on), "
    #     "which is the chronological timeline of events leading to the two companies' merger. "
    #     "With that section, determine who was the first entity to **actively initiate or facilitate** discussions leading to the merger/deal. "
    #     "This could be one of the companies involved or a third party, such as an investor, advisor, consultant, or financial institution. "
    #     "If an external entity introduced the two companies or arranged the initial discussions, return their name instead. "
    #     "Provide only the initiator's name in the exact format with brackets so I can separate them later: [initiator name]."
    # );

    prompt = (
        "Locate the 'Background of the Merger' section (which may be titled differently, such as "
        "'Background of the Transaction', 'Background of the Acquisition', or 'Background of the Offer'). "
        "This section provides a chronological timeline of events leading to the merger. "
        "Identify the first entity that **actively initiated or facilitated** discussions leading to the merger or acquisition. "
        "The initiator must be a company or financial institution, **not an individual person**. "
        "Read the beginning of the document if the initiator is an individual or an entity that is not a company. "
        "If a third-party advisor, consultant, or financial institution arranged the initial discussions, return their name instead. "
        "Strictly return only the initiating company's name within brackets: [initiator name]. "
    );

    assistant = Assistant(api_key, prompt, "gpt-4o-mini");

    crawler = Crawler(filedDate, companyAList, companyBList, startPhrases, maxNumOfThreads, nlp, assistant);
    crawler.runCrawler(index=6);

    if deleteAssistant:
        assistant.deleteAssistant();


if __name__ == "__main__":
    main();