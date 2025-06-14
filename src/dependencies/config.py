import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv();

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY");

# Program configuration variables
MAX_NUM_OF_THREADS = min(32, os.cpu_count() + 4);

# MongoDB configuration
MONGO_URL = "mongodb://localhost:27017";
DATASET_NAME = "DataSet";
EXTRACTEDSECTIONS_NAME = "ExtractedSections";

# A more lightweight SpaCy model to quickly discern the section (can be replaced with larger models)
BASE_NLP_MODEL = "en_core_web_sm"; # String format to be instantiated in each process generated via multi-processing
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3";
EMBEDDING_MODEL = "text-embedding-3-small";
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
    "Background of the Open Market Merger",
    "Background"
];

# Configuration for similarity search and re-ranking
QUERY_EMBEDDING_FILE = os.path.abspath("./config/query_embedding.json");
RERANK_QUERY_FILE = os.path.abspath("./config/rerank_query.txt");
BATCH_SIZE = 128; # Computing power specific. Tune for your device.

LOG_FILE_PATH = os.path.abspath("./logs.txt");

# Fallback tools and queries
FALLBACK_MODEL = "o4-mini-2025-04-16";
FALLBACK_TOOLS = [{
    "type": "function",
    "function": {
        "name": "determine_background_section",
        "description": (
            "Analyze the given text and determine if a section exists that serves the same role as the "
            "'background of the merger' section. This section may appear under alternative titles, including but not limited to "
            "'Background of the Transaction', 'Background of the Acquisition', 'Background of the Offer', "
            "'Background of the Consolidation', or simply 'Background'.\n\n"

            "The 'background of the merger' section typically provides a **chronological narrative** of the events, meetings, decisions, "
            "and negotiations that led to the merger agreement. It often includes specific **dates**, names of executives or companies, and summaries "
            "of board discussions or strategic developments. Valid sections include detailed timelines or multiple steps of the decision-making process.\n\n"

            "Do **not** return a match if the phrase appears only in:\n"
            "- citations or references (e.g., 'see Background of the Offer')\n"
            "- table of contents or index\n"
            "- legal language amending or incorporating a section by reference\n"
            "- summary lists of document sections\n\n"

            "**Important:**\n"
            "- Only return `hasSection: true` if the narrative section is clearly present and detailed.\n"
            "- If `hasSection` is `false`, you may omit `matchHeader` or leave it empty.\n"
            "- `confidence` is optional when `hasSection` is false — if used, a low score (e.g., < 0.4) is appropriate.\n\n"

            "**Examples:**\n\n"
            "Valid example (real background section):\n"
            "Background of the Merger\n\n"
            "    During the last several years, Mediconsult has held conversations with a number of companies to evaluate possible business combinations...\n"
            "    Beginning the first week in September 2000, Ian Sutcliffe, Chief Executive Officer of Mediconsult, had several telephone conversations...\n"
            "    On October 4, 2000, the Mediconsult board held a special meeting to discuss possible strategic transactions.\n\n"

            "**Your task is to return:**\n"
            "- hasSection: True or False\n"
            "- matchHeader: the exact header string only, if applicable\n"
            "- confidence: your certainty score (0 to 1), only if applicable"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hasSection": {
                    "type": "boolean",
                    "description": (
                        "Whether the 'background of the merger' section is present in the given text. "
                        "Return False if it only appears in citations, summaries, or other indirect references."
                    )
                },
                "matchHeader": {
                    "type": "string",
                    "description": (
                        "The exact header or phrase matched (e.g., 'Background of the Offer'). "
                        "Leave blank or omit if `hasSection` is false."
                    )
                },
                "confidence": {
                    "type": "number",
                    "description": (
                        "A confidence score between 0 and 1 indicating how certain the model is. "
                        "Optional if `hasSection` is false."
                    )
                }
            },
            "required": [
                "hasSection"
            ],
            "additionalProperties": False
        }
    }
}];

# Initiator identifier tools and queries
IDENTIFIER_MODEL = "o4-mini-2025-04-16";
IDENTIFIER_TOOLS = [{
    "type": "function",
    "function": {
        "name": "identify_initiator",
        "description": (
            "Analyze the provided merger-related text to determine who initiated the deal and why. "
            "Classify the initiation as one of four types:\n"
            "- 'Acquirer-Initiated Deal' — the acquiring company proposed the transaction\n"
            "- 'Target-Initiated Deal' — the company being acquired proposed the transaction\n"
            "- 'Third-Party-Initiated Deal' — an external party such as an investor, advisor, or regulator initiated the process\n"
            "- 'Mutual' — both companies jointly pursued the deal without a clearly dominant initiator\n\n"

            "Extract the full legal name of the initiating entity (avoid vague references like 'Offeror' or 'Parent' unless unavoidable), "
            "the approximate date of the first meaningful contact, and the primary reasons cited for the merger.\n\n"

            "Also resolve ambiguous initiator names (e.g., Offeror) to the parent company if possible, and expand abbreviated company names. "
            "Note that the potential abbreviations for company names will be provided."
            "Do also note any conflicting or contradictory sources, and provide reasoned interpretations when ambiguity exists.\n\n"

            "Remain neutral and factual—avoid speculation."

            "**The information accuracy is important in determining whether price manipulation is present depending on who is initiating the merger.** \n\n"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "initiator": {
                    "type": "string",
                    "description": (
                        "The full legal name of the company that first initiated or proposed the merger."
                        "If the deal was mutually initiated, return 'Mutual'. "
                        "If the deal was initiated by a third party, return 'Third-Party'. "
                        "Avoid vague terms like 'Parent' or 'Offeror' unless no other information is available."
                    )
                },
                "date_of_initiation": {
                    "type": "string",
                    "description": (
                        "The date of first meaningful contact between the companies or their representatives regarding the merger or acquisition." 
                        "This includes emails, meetings, letters of intent, or informal discussions leading to a formal proposal."
                    )
                },
                "type_of_initiation": {
                    "type": "string",
                    "enum": [
                        "Acquirer-Initiated Deal",
                        "Target-Initiated Deal",
                        "Third-Party-Initiated Deal",
                        "Mutual"
                    ],
                    "description": "The classification of who initiated the deal: the acquiring company, the target company, a third party (such as an investor or advisor arranging the merger), or both parties jointly."
                },
                "stated_reasons": {
                    "type": "string",
                    "description": "A concise summary of the stated or implied motivations for the merger, including strategic, financial, legal, or operational factors."
                }
            },
            "required": [
                "initiator",
                "date_of_initiation",
                "type_of_initiation",
                "stated_reasons"
            ],
            "additionalProperties": False
        }
    }
}];