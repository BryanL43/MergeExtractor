import pandas as pd
from dotenv import load_dotenv
import os
import spacy
import torch

from Assistant import Assistant
from Crawler import Crawler
from Cognition import Cognition

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
    "Background of the Offer and the Merger",
    "Background to the merger",
    "Background to the acquisition",
    "Background to the offer",
    "Background to the transaction"
];

def main():
    # Extract the documents with both company names present and the "Background of the Merger" section
    instructions = (
        "You specialize in locating and extracting relevant information from a specific section of a given text file. "
        "Your task is to identify the relevant section, analyze its content, and then respond to the given prompt based on your analysis. "
        "Make sure you have gathered all content from the section by checking for any possible amendments or additions. "
        "If you cannot find the section, simply return 'None'."
    );
    
    prompt = (
        "Locate the 'Background of the Merger' section (which may be titled differently, such as "
        "'Background of the Transaction', 'Background of the Acquisition', or 'Background of the Offer'). "
        "This section provides a chronological timeline of events leading to the merger. "
        "For example, the timeline will describe events like: On date X, company A met with company B. "
        "If and only if you find this section, strictly return [Found]. "
        "If you cannot find the section, strictly return [None]. Do not return anything else. "
    );
    
    filterAssistant = Assistant(api_key, "Filter Assistant", instructions, prompt, "gpt-4o-mini");

    crawler = Crawler(filedDate, companyAList, companyBList, startPhrases, maxNumOfThreads, nlp, filterAssistant);
    # crawler.runCrawler(startIndex=0, endIndex=20); # True to literal index: i.e., 0 to 99 is 0 to 99
    # crawler.runCrawler(index=20);

    # Find the company that had the intention of selling/buying the other company
    instructions = (
        "This is a test"
    );
    prompt = (
        "This is a test"
    );
    analystAssistant = Assistant(api_key, "Analyst Assistant", instructions, prompt, "gpt-4o-mini");

    # cognition = Cognition(companyAList, companyBList, 5, analystAssistant); # 5 threads to not flood openai api
    # cognition.findInitiator(index=0); # Index literal; 0 is 0

    if deleteAssistant:
        filterAssistant.deleteAssistant();
        analystAssistant.deleteAssistant();


if __name__ == "__main__":
    main();