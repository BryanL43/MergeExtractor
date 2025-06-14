from src.identifier.InitiatorIdentifier import InitiatorIdentifier

def main():
    identifier = InitiatorIdentifier();
    identifier.runIdentifier(index=76);
    # identifier.runIdentifier(start_index=0, end_index=5, batch_size=3);


if __name__ == "__main__":
    main();
