from src.seperator.SeperatorHandler import SeperatorHandler

def main():
    seperator = SeperatorHandler();
    # seperator.runSeperator(index=560);
    seperator.runSeperator(start_index=900, end_index=999, batch_size=3);


if __name__ == "__main__":
    main();
