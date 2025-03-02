import pandas as pd
from dotenv import load_dotenv
import os
import spacy
import torch

from FileAnalyzerAssistant import FileAnalyzerAssistant
from TextAnalyzerAssistant import TextAnalyzerAssistant
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
        "The assistant is tasked with processing a large set of documents to determine who initiated a merger deal and the reason behind it. "
        "The assistant will extract relevant details, summarize key points, and analyze the motivation for the merger."

        "Core Capabilities:"
        "Document Ingestion & Parsing"

        "Accept and process multiple document formats (PDF, Word, text, etc.)."
        "Extract structured text while preserving relevant context."
        "Merger Initiation Identification"

        "Identify the company, individual, or entity that first proposed the merger."
        "Extract the date of initiation and any key meetings or correspondence leading to the proposal."
        "Identify key decision-makers (CEOs, board members, investors, etc.)."
        "Reason Analysis"

        "Extract and summarize the stated reasons for the merger, including:"
        "Financial struggles or growth opportunities."
        "Market expansion, competitive positioning, or strategic benefits."
        "Regulatory or legal pressures."
        "Internal motivations (e.g., shareholder demands, leadership changes)."
        "If multiple reasons are given, determine the most cited justification."
        "Contradictions & Conflicts"

        "Identify conflicting reports about who initiated the deal."
        "Highlight discrepancies between internal documents, news reports, and official statements."
        "Provide possible explanations for conflicting narratives."
        "Summarization & Reporting"

        "Deliver a structured summary including:"
        "Initiating Company: Who first proposed the merger."
        "Date of Initiation: When the proposal was made."
        "Stated Reason(s): Why the merger was proposed."
        "Key Figures Involved: Executives, stakeholders, or board members."
        "Output in structured formats (text summary, bullet points, tables, JSON, or reports)."

        "Processing Guidelines:"
        "Prioritize factual accuracy when determining the initiating company."
        "Cross-reference multiple sources if available to verify consistency."
        "If the initiation is unclear, list possible candidates and the evidence supporting each."
        "Maintain neutrality and avoid speculative conclusions."

        "Example Queries:"
        "Who was the first company to initiate the merger and why?"
        "Summarize all reasons provided for the merger."
        "Compare internal vs. external narratives about the merger's origin."
    );

    query = (
        "Locate the 'Background' section of the merger agreement. "
        "The section is a chronological timeline of events leading up to the merger. Who initiated the merger agreement? "
    );

    analystAssistant = TextAnalyzerAssistant(api_key, "Analyst Assistant", instructions, query, "gpt-4o-mini");

    cognition = Cognition(companyAList, companyBList, analystAssistant, threadCount=5, chunkSize=20, modelName="all-MiniLM-L6-v2");
    # cognition.findInitiator(startIndex=0, endIndex=19); # Index literal; 0 is 0
    cognition.findInitiator(index=6);

    if deleteAssistant:
        filterAssistant.deleteAssistant();
        # analystAssistant.deleteAssistant();


if __name__ == "__main__":
    main();