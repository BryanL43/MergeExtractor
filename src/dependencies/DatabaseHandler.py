from pymongo import MongoClient
from src.dependencies.config import MONGO_URL, DATASET_NAME, EXTRACTEDSECTIONS_NAME

class DatabaseHandler:
    def __init__(self):
        self.client = MongoClient(MONGO_URL);
        self.dataset_db = self._get_or_init_db(DATASET_NAME);
        self.extracted_sections_db = self._get_or_init_db(EXTRACTEDSECTIONS_NAME);

        self._ensure_batch_collections(self.dataset_db);
        self._ensure_batch_collections(self.extracted_sections_db);

    def _get_or_init_db(self, db_name: str):
        if not db_name in self.client.list_database_names():
            print(f"[!] Database '{db_name}' does not exist yet. Initializing...");
            # Force create by inserting a dummy doc and then removing it
            temp_db = self.client[db_name];
            temp_db["__init__"].insert_one({"init": True});
            temp_db.drop_collection("__init__");
            print(f"[+] Database '{db_name}' created.");

        return self.client[db_name];

    def _ensure_batch_collections(self, db):
        """Ensures batch collections exist for the given database."""
        existing_collections = db.list_collection_names();

        for batch_start in range(0, 1800, 100):
            batch_end = batch_start + 99;
            collection_name = f"batch_{batch_start}_{batch_end}";

            if collection_name not in existing_collections:
                print(f"[+] Creating collection: {collection_name} in {db.name}");
                db.create_collection(collection_name);

    def __enter__(self):
        return self;

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.close();
