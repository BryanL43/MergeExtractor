import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv();

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY");

# Program configuration variables
MAX_NUM_OF_THREADS = min(32, os.cpu_count() + 4);

# A more lightweight SpaCy model to quickly discern the section (can be replaced with larger models)
BASE_NLP_MODEL = "en_core_web_sm"; # String format to be instantiated in each process generated via multi-processing
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3";
DELETE_ASSISTANT_MODE = False;
CSV_FILE = os.path.abspath("./truncatedData.csv");

# Read the CSV file and extract the date & both merging companies (index base)
ANNOUNCEMENT_DATES = pd.read_csv(CSV_FILE, header=None).iloc[:, 1].tolist();
COMPANY_A_LIST = pd.read_csv(CSV_FILE, header=None).iloc[:, 2].tolist();
COMPANY_B_LIST = pd.read_csv(CSV_FILE, header=None).iloc[:, 3].tolist();

FORM_TYPES = ["PREM14A", "S-4", "SC 14D9", "SC TO-T"];

# Potential candidates for the target section title
START_PHRASES = [
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

# Configuration for similarity search and re-ranking
QUERY_EMBEDDING_FILE = os.path.abspath("./config/query_embedding.json");
RERANK_QUERY_FILE = os.path.abspath("./config/rerank_query.txt");
BATCH_SIZE = 128; # Computing power specific. Tune for your device.

LOG_FILE_PATH = os.path.abspath("./logs.txt");