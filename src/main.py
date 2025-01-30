# Main imports
from dotenv import load_dotenv
import os

# Object imports
from Assistant import Assistant

def main():
    load_dotenv();
    api_key = os.getenv("OPENAI_API_KEY");

    prompt = (
        "Extract the 'Background of the Merger'"
        "(which could be phrased differently, such as 'Background of the Transaction', 'Background of the Acquisition', or 'Background of the Offer', and so on),"
        "which is the chronological timeline of events leading to the two companies' merger."
        "With that extracted section, could you tell me who initiated the merger/deal first."
    );

    myAssistant = Assistant(api_key, prompt, "gpt-4o-mini");
    myAssistant.deleteAssistant();


if __name__ == "__main__":
    main();