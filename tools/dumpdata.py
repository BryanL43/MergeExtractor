import os
from src.dependencies.DatabaseHandler import DatabaseHandler

def dump_collection_to_filesystem(db_ref, output_base_path: str, db_label: str):
    for batch_start in range(0, 1800, 100):
        batch_end = batch_start + 99;
        collection_name = f"batch_{batch_start}_{batch_end}";
        output_dir = f"{output_base_path}/{batch_start}-{batch_end}";
        os.makedirs(output_dir, exist_ok=True);

        collection = db_ref[collection_name];
        documents = collection.find();

        for doc in documents:
            main_index = doc.get("main_index");
            company_A = doc.get("company_A", "").replace(" ", "_");
            company_B = doc.get("company_B", "").replace(" ", "_");
            content = doc.get("content", "");
            format_doc_name = f"{main_index}_{company_A}_&_{company_B}.txt";
            file_path = os.path.join(output_dir, format_doc_name);

            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    url = doc.get("url");
                    if url:
                        f.write(f"URL: {url}\n\n");

                    f.write(content);
                print(f"[{db_label}] Dumped: {format_doc_name}");
            except Exception as e:
                print(f"[{db_label}] Failed to write {file_path}: {e}");

def dump_mongodb_to_filesystem():
    with DatabaseHandler() as db:
        dump_collection_to_filesystem(db.dataset_db, "./DataSet", "DataSet");
        dump_collection_to_filesystem(db.extracted_sections_db, "./ExtractedSections", "ExtractedSections");

    print("MongoDB to filesystem dump complete.");

if __name__ == "__main__":
    dump_mongodb_to_filesystem();
