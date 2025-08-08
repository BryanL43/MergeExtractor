from src.seperator.SeperatorHandler import SeperatorHandler

def main():
    seperator = SeperatorHandler();
    # seperator.runSeperator(index=560);
    seperator.runSeperator(start_index=1700, end_index=1701, batch_size=2);


if __name__ == "__main__":
    main();
