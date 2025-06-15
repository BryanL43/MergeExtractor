from src.seperator.SeperatorHandler import SeperatorHandler

def main():
    seperator = SeperatorHandler();
    seperator.runSeperator(index=443);
    # seperator.runSeperator(start_index=400, end_index=499, batch_size=3);


if __name__ == "__main__":
    main();
