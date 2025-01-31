# Main imports
import pandas as pd
from dotenv import load_dotenv
import os

# Object imports
from Assistant import Assistant
from Crawler import Crawler

load_dotenv();

# Global variables
api_key = os.getenv("OPENAI_API_KEY");
csvFile = "./truncatedData.csv";
maxNumOfThreads = os.cpu_count();

# Read the CSV file and extract the date & both merging companies (index base)
filedDate = pd.read_csv(csvFile, header=None).iloc[:, 1].tolist();
companyAList = pd.read_csv(csvFile, header=None).iloc[:, 2].tolist();
companyBList = pd.read_csv(csvFile, header=None).iloc[:, 3].tolist();

def main():
    prompt = (
        "Extract the 'Background of the Merger'"
        "(which could be phrased differently, such as 'Background of the Transaction', 'Background of the Acquisition', or 'Background of the Offer', and so on),"
        "which is the chronological timeline of events leading to the two companies' merger."
        "With that extracted section, could you tell me who initiated the merger/deal first."
    );

    assistant = Assistant(api_key, prompt, "gpt-4o-mini");
    # print(myAssistant.extractSection("./DataSet/Xircom_Inc_&_Intel_Corp.txt").split("---")[1]);
    # myAssistant.deleteAssistant();

    crawler = Crawler(filedDate, companyAList, companyBList, maxNumOfThreads, assistant);
    crawler.runCrawler(index=11);


if __name__ == "__main__":
    main();