from openai import OpenAI
import json
import os
import sys
import time

# I only need a single assistant, so I'm limiting the ability to create multiple.
class Assistant:
    def __init__(self, api_key: str, name: str, instructions: str, prompt: str, model: str, temp: float = 1.00, top_p: float = 1.00):
        self.__client = OpenAI(api_key=api_key);
        self.__name = name;
        self.prompt = prompt;
        self.__instructions = instructions;
        self.__model = model;
        self.__temp = temp;
        self.__top_p = top_p;
        
        self.__assistant_id = None;

        # Load existing assistants from JSON
        self.__assistants_data = self.__loadAssistants();

        # Check if an assistant with the same name already exists
        for assistant in self.__assistants_data:
            if assistant["name"] == self.__name:
                self.__assistant_id = assistant["id"];
                break;
        
        # If assistant was not found, create a new one
        if not self.__assistant_id:
            self.__createAssistant();
        
        # Validate assistant existence
        try:
            self.__client.beta.assistants.retrieve(self.__assistant_id);
        except Exception as e:
            print(f"Automatic correction: {self.__name} assistant not found. Creating a new one...");
            self.__createAssistant();
    
        print(f"Assistant initialized: {self.__name}");
    
    def __loadAssistants(self):
        """Load assistants from the JSON file."""
        # start_time = time.time()

        if os.path.exists("assistantData.json"):
            try:
                with open("assistantData.json", "r") as json_file:
                    return json.load(json_file) or [];
            except json.JSONDecodeError:
                return [];
        return [];
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Elapsed time: {elapsed_time} seconds")

    
    def __createAssistant(self):
        self.__assistant = self.__client.beta.assistants.create(
            name=self.__name,
            instructions = self.__instructions,
            model=self.__model,
            tools=[{"type": "file_search"}],
            temperature=self.__temp,
            top_p = self.__top_p
        );
    
        self.__assistant_id = self.__assistant.id;
    
        # Check if an assistant with the same name exists
        existing_assistant = next((a for a in self.__assistants_data if a["name"] == self.__name), None);

        if existing_assistant:
            # Overwrite the existing assistant's ID
            existing_assistant["id"] = self.__assistant_id;
        else:
            # Add new assistant if no match is found
            self.__assistants_data.append({
                "id": self.__assistant_id,
                "name": self.__name
            });

        # Save the assistant data
        with open("assistantData.json", "w") as json_file:
            json.dump(self.__assistants_data, json_file, indent=4);
    
        print(f"Successfully created Assistant: {self.__name}");

    # Find the "Background of the Merger" section
    def analyzeDocument(self, file_path: str):
        start_time = time.time()  # Start timing
        msg_file = self.__client.files.create(
            file=open(file_path, "rb"),
            purpose="assistants"
        );

        thread = self.__client.beta.threads.create(
            messages = [
                {
                    "role": "user",
                    "content": self.prompt,
                    "attachments": [
                        {
                            "file_id": msg_file.id,
                            "tools": [{"type": "file_search"}]
                        },
                    ]
                }
            ]
        );

        run = self.__client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=self.__assistant_id
        )

        messages = list(self.__client.beta.threads.messages.list(
            thread_id=thread.id,
            run_id=run.id
        ));

        result = messages[0].content[0].text.value;

        # Delete the thread, file, & message but not the assistant (can reuse)
        self.__client.files.delete(msg_file.id);
        self.__client.beta.threads.delete(thread_id=thread.id);

        if run.status == "completed":
            end_time = time.time()  # End timing
            elapsed_time = end_time - start_time
            print(f"Time taken to analyze document: {elapsed_time:.4f} seconds")
            return result;
        else:
            raise RuntimeError(f"Assistant failed with status: {run.status}");

    def clearVectorStores(self):
        # Since we are using thread directly, the vector is created automatically.
        # API doesn't specify how to acquire a specific one, so I'm just flushing it out.
        vector_stores = self.__client.beta.vector_stores.list();
        for vector in vector_stores.data:
            self.__client.beta.vector_stores.delete(vector_store_id=str(vector.id));

    def deleteAssistant(self):
        self.__client.beta.assistants.delete(self.__assistant_id);

        if os.path.exists("assistantData.json"):
            with open("assistantData.json", "r") as json_file:
                self.__assistants_data = json.load(json_file);

            # Filter out the assistant to be deleted
            self.__assistants_data = [a for a in self.__assistants_data if a["id"] != self.__assistant_id];

            # Update the JSON file or remove it if empty
            if self.__assistants_data:
                with open("assistantData.json", "w") as json_file:
                    json.dump(self.__assistants_data, json_file, indent=4);
            else:
                os.remove("assistantData.json");

        print(f"Successfully deleted Assistant: {self.__name}");