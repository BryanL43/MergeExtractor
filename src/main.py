import pandas as pd
from dotenv import load_dotenv
import os
import spacy
from concurrent.futures import ThreadPoolExecutor

from BackupAssistant import BackupAssistant
from AnalysisAssistant import AnalysisAssistant
from Crawler import Crawler
from InitiatorClassifier import InitiatorClassifier

# Config variables
MAX_NUM_OF_THREADS = min(32, os.cpu_count() + 4); # From docs
DELETE_ASSISTANT_MODE = False;
CSV_FILE = "./truncatedData.csv";

def main():
    local = load_dotenv();
    if not local:
        raise RuntimeError("Environment variables not loaded. Please check your .env file.");

    openai_api_key = os.getenv("OPENAI_API_KEY");
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env file.");

    nlp = spacy.load("en_core_web_sm");

    # Read the CSV file and extract the date & both merging companies (index base)
    filed_date = pd.read_csv(CSV_FILE, header=None).iloc[:, 1].tolist();
    company_A_list = pd.read_csv(CSV_FILE, header=None).iloc[:, 2].tolist();
    company_B_list = pd.read_csv(CSV_FILE, header=None).iloc[:, 3].tolist();

    # Phrases for locating start point of the background section
    start_phrases = [
        "Background of the transaction",
        "Background of the merger",
        "Background of the offer",
        "Background of the acquisition",
        "Background of the consolidation",
        "Background of the Asset Sale",
        "Background of the Combination",
        "Background of the Proposal",
        "Background of the Offer and the Merger",
        "Background and negotiation of the merger",
        "Background to the merger",
        "Background to the acquisition",
        "Background to the offer",
        "Background to the transaction",
        "Background to the consolidation",
        "Background to the Asset Sale",
        "Background to the Combination",
        "Background to the Proposal",
        "Background of Offer",
        "Background of Acquisition",
        "Background of Transaction",
        "Background of Merger",
        "Background of Consolidation",
        "Background of Asset Sale",
        "Background of Combination",
        "Background of Proposal",
        "Background of the Proposed Transaction",
        "Background"
    ];

    # Instantiating the shared thread pool for shared usage and automatic handling
    thread_pool = ThreadPoolExecutor(max_workers=MAX_NUM_OF_THREADS);
    
    backup_assistant = BackupAssistant(openai_api_key, "Backup Assistant", "gpt-4o-mini");

    crawler = Crawler(filed_date, company_A_list, company_B_list, start_phrases, thread_pool, nlp, backup_assistant);
    crawler.runCrawler(index=192, date_margin=4);
    # crawler.runCrawler(start_index=0, end_index=49, date_margin=4);

    analysisAssistant = AnalysisAssistant(openai_api_key, "Analysis Assistant", "gpt-4o-mini");

    initiatorClassifier = InitiatorClassifier(openai_api_key, company_A_list, company_B_list, start_phrases, thread_pool, nlp, analysisAssistant);
    # initiatorClassifier.findInitiator(index=91);
    # initiatorClassifier.findInitiator(start_index=0, end_index=49);

    thread_pool.shutdown(wait=True);

    if DELETE_ASSISTANT_MODE:
        print("Deleting assistants...");
        backup_assistant.deleteAssistant();
        initiatorClassifier.deleteAssistant();


if __name__ == "__main__":
    main();