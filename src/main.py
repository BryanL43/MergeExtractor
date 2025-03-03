import pandas as pd
from dotenv import load_dotenv
import os
import spacy
import torch

from FileAnalyzerAssistant import FileAnalyzerAssistant
from ContextAssistant import ContextAssistant
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
    "Background of Proposal"
];

def main():
    # Extract the documents with both company names present and the "Background of the Merger" section
    instructions = (
        "Your primary task is to determine whether the given text contains a section that provides a chronological background of a merger. "
        "This section may be titled differently, such as 'Background of the Transaction', 'Background of the Acquisition', 'Background of the Offer', or 'Background of the Consolidation' and so on. "
        "Carefully scan the document to check if such a section exists. "
        "Do not analyze the contents—only confirm whether the section is present. "
        "If the section is found, return [Found]. If not, return [None]."
    );
    
    query = (
        "Locate the 'Background of the Merger' section (which may be titled differently, such as "
        "'Background of the Transaction', 'Background of the Acquisition', 'Background of the Consolidation', or 'Background of the Offer'). "
        "This section provides a chronological timeline of events leading to the merger. "
        "For example, the timeline will describe events like: On date X, company A met with company B. "
        "If and only if you find this section, strictly return [Found]. "
        "If you cannot find the section, strictly return [None]. Do not return anything else. "
    );
    
    filterAssistant = FileAnalyzerAssistant(api_key, "Filter Assistant", instructions, query, "gpt-4o-mini");

    crawler = Crawler(filedDate, companyAList, companyBList, startPhrases, maxNumOfThreads, nlp, filterAssistant);
    # crawler.runCrawler(startIndex=250, endIndex=299); # True to literal index: i.e., 0 to 99 is 0 to 99
    # crawler.runCrawler(index=300);

    # Find the company that had the intention of selling/buying the other company
    instructions = (
        "The assistant is tasked with processing a large set of documents to determine who initiated a merger deal and the reason behind it.\n"
        "The assistant will extract relevant details, summarize key points, and analyze the motivation for the merger.\n\n"

        "Core Capabilities:\n"
        "Document Ingestion & Parsing\n"
        "Accept and process multiple document formats (PDF, Word, text, etc.).\n"
        "Extract structured text while preserving relevant context.\n\n"

        "Merger Initiation Identification\n"
        "Identify the company, individual, or entity that first proposed the merger.\n"
        "Extract the date of initiation and any key meetings or correspondence leading to the proposal.\n"
        "Identify key decision-makers (CEOs, board members, investors, etc.).\n\n"

        "Handling Parent & Subsidiary Companies\n"
        "Resolve cases where the merger is initiated through a subsidiary (e.g., Offeror) on behalf of a parent company.\n"
        "If an entity like 'Offeror' or a similar term is used, verify whether it is acting on behalf of another company.\n"
        "Explicitly state the full legal name of the **actual** initiating entity, avoiding vague terms like 'Parent' or 'Offeror' unless no further information is provided.\n\n"

        "Abbreviation & Name Resolution\n"
        "Expand company abbreviations and acronyms when possible to ensure clarity.\n"
        "Cross-reference context to distinguish between common names and corporate entities (e.g., 'GE' as 'General Electric').\n"
        "If ambiguity exists, provide a list of possible interpretations along with supporting evidence from the text.\n\n"

        "Reason Analysis\n"
        "Extract and summarize the stated reasons for the merger, including:\n"
        "Financial struggles or growth opportunities.\n"
        "Market expansion, competitive positioning, or strategic benefits.\n"
        "Regulatory or legal pressures.\n"
        "Internal motivations (e.g., shareholder demands, leadership changes).\n"
        "If multiple reasons are given, determine the most cited justification.\n\n"

        "Contradictions & Conflicts\n"
        "Identify conflicting reports about who initiated the deal.\n"
        "Highlight discrepancies between internal documents, news reports, and official statements.\n"
        "Provide possible explanations for conflicting narratives.\n\n"

        "Summarization & Reporting\n"
        "Deliver a structured summary including:\n"
        "Initiating Company: Who first proposed the merger.\n"
        "Date of Initiation: When the proposal was made.\n"
        "Stated Reason(s): Why the merger was proposed.\n"
        "Key Figures Involved: Executives, stakeholders, or board members.\n"
        "Output in structured formats (text summary, bullet points, tables, JSON, or reports).\n\n"

        "Processing Guidelines:\n"
        "Prioritize factual accuracy when determining the initiating company.\n"
        "Cross-reference multiple sources if available to verify consistency.\n"
        "If the initiation is unclear, list possible candidates and the evidence supporting each.\n"
        "Maintain neutrality and avoid speculative conclusions.\n\n"

        "Example Queries:\n"
        "Who was the first company to initiate the merger and why?\n"
        "Summarize all reasons provided for the merger.\n"
        "Compare internal vs. external narratives about the merger's origin.\n"
    );

    query = (
        "Locate the 'Background' section of the merger agreement. "
        "The section is a chronological timeline of events leading up to the merger. "
        "Who initiated the merger agreement? "
    );

    analystAssistant = ContextAssistant(api_key, "Analyst Assistant", instructions, query, "gpt-4o-mini");

    cognition = Cognition(companyAList, companyBList, analystAssistant, threadCount=5);
    # cognition.findInitiator(startIndex=0, endIndex=10); # Index literal; 0 is 0
    cognition.findInitiator(index=3);

    if deleteAssistant:
        # filterAssistant.deleteAssistant();
        analystAssistant.deleteAssistant();


if __name__ == "__main__":
    main();