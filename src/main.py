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
    "Background to the transaction",
    "Background of Offer",
    "Background of Acquisition",
    "Background of Transaction",
    "Background of Merger"
];

def main():
    # Extract the documents with both company names present and the "Background of the Merger" section
    instructions = (
        "Your primary task is to determine whether the given text contains a section that provides a chronological background of a merger. "
        "This section may be titled differently, such as 'Background of the Transaction', 'Background of the Acquisition', or 'Background of the Offer', and so on. "
        "Carefully scan the document to check if such a section exists. "
        "Do not analyze the contents—only confirm whether the section is present. "
        "If the section is found, return [Found]. If not, return [None]."
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
    # crawler.runCrawler(startIndex=21, endIndex=49); # True to literal index: i.e., 0 to 99 is 0 to 99
    crawler.runCrawler(index=0);

    # Find the company that had the intention of selling/buying the other company
    instructions = (
        "You specialize in identifying the company that first expressed an intention to sell or buy the other company in a given text file. "
        "Your primary task is to locate the relevant section that outlines the timeline of merger discussions, acquisitions, or sell-off considerations. "
        "If a third party (such as an advisor, investor, or regulatory body) initiated the suggestion, extract the company that ultimately acted upon it. "
        "Ensure that you analyze the sequence of events carefully to determine which company first demonstrated intent—whether by initiating discussions, hiring advisors, or making a formal proposal. "
        "If you identify this company, return its name strictly in the format: [Company Name]. If you cannot determine the initiating company, return [None]."
    );

    prompt = (
        "Analyze the document to determine which company first expressed intent to either sell itself or acquire another company. "
        "Look for initial merger discussions, acquisition proposals, or sell-off considerations. "
        "If a third party (e.g., an advisor or investor) suggested the deal, extract the company that took action on it. "
        "Return the initiating company's name strictly in the format: [Company Name]. "
        "If no clear initiating company is found, return [None]. "
        "Do return why said company is the first to express intent. "
    );

    # analystAssistant = Assistant(api_key, "Analyst Assistant", instructions, prompt, "gpt-4o-mini");

    # cognition = Cognition(companyAList, companyBList, 5, analystAssistant); # 5 threads to not flood openai api
    # cognition.findInitiator(startIndex=0, endIndex=19); # Index literal; 0 is 0
    # cognition.findInitiator(index=0);

    if deleteAssistant:
        filterAssistant.deleteAssistant();
        # analystAssistant.deleteAssistant();


if __name__ == "__main__":
    main();