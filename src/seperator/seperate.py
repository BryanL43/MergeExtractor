from src.seperator.SeperatorHandler import SeperatorHandler

def main():
    seperator = SeperatorHandler();
    # seperator.runSeperator(index=429);
    seperator.runSeperator(start_index=450, end_index=499, batch_size=3);


if __name__ == "__main__":
    main();
