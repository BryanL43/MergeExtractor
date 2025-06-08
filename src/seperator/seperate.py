from src.seperator.SeperatorHandler import SeperatorHandler

def main():
    seperator = SeperatorHandler();
    # seperator.runSeperator(index=16);
    seperator.runSeperator(start_index=2, end_index=10, batch_size=3);


if __name__ == "__main__":
    main();

