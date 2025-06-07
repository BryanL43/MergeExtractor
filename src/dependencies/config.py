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

# Fallback tools and queries
FALLBACK_MODEL = "gpt-4o-mini";
FALLBACK_TOOLS = [{
    "type": "function",
    "function": {
        "name": "determine_background_section",
        "description": (
            "Analyze the given text and determine if a section exists that serves the same role as the "
            "'background of the merger' section. This section may appear under alternative titles, including but not limited to "
            "'Background of the Transaction', 'Background of the Acquisition', 'Background of the Offer', "
            "'Background of the Consolidation', or simply 'Background'. You should infer the presence of the section "
            "based on contextual and structural cues, even if the exact phrase is not used.\n\n"

            "The 'background of the merger' section typically provides a **chronological narrative** of the events, meetings, decisions, "
            "and negotiations that led to the merger agreement. It often includes **dates**, names of involved companies, and summaries "
            "of board discussions or strategic rationales.\n\n"

            "Be cautious of false positives. Do not match if the phrase appears only in citations, references (e.g., 'see Background of the Merger'), "
            "table of contents, or mentions without section content.\n\n"

            "**Your task is to identify whether such a section is truly present, and return:**\n"
            "- hasSection: True or False\n"
            "- matchHeader: the exact header string only (e.g., 'Background', or 'Background of the Offer')\n"
            "- confidence: your certainty score (0 to 1)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hasSection": {
                    "type": "boolean",
                    "description": "Whether the 'background of the merger' section is present in the given text."
                },
                "matchHeader": {
                    "type": "string",
                    "description": "The exact header or phrase matched (if any). Reply with the exact header string only."
                },
                "confidence": {
                    "type": "number",
                    "description": "A confidence score between 0 and 1 indicating how certain the model is"
                }
            },
            "required": [
                "hasSection",
                "matchHeader",
                "confidence"
            ],
            "additionalProperties": False
        }
    }
}];