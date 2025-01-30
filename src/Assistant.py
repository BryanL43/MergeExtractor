from openai import OpenAI
import json
import os

# I only need a single assistant, so I'm limiting the ability to create multiple.
class Assistant:
    def __init__(self, api_key: str, prompt: str, model: str, temp: float = 1.00, top_p: float = 1.00):
        self.__client = OpenAI(api_key=api_key);
        self.prompt = prompt;
        self.model = model;
        self.temp = temp;
        self.top_p = top_p;
    
        # Create the assistant with limited parameters
        if os.path.exists("assistantData.json"):
            self.__assistant = self.__client.beta.assistants.create(
                name="Document Assistant",
                instructions="""\
                    You are an expert at identifying and extracting a section from a given text file.
                    You are to locate the desired section and return its entire contents.
                    Please give me the section exactly as it is without summarizing, paraphrasing, or changing anything.
                    Ensure that you have acquired everything within the section by verifying any additional amendments.
                    If you cannot find the section, simply return 'None'.
                    """,
                model=self.model,
                tools=[{"type": "file_search"}],
                temperature=self.temp,
                top_p = self.top_p
            );

        # Write the dictionary to a JSON file
        data = {
            "id": self.__assistant.id
        };

        with open("assistantData.json", "w") as json_file:
            json.dump(data, json_file);
    
        print("Successfully created Assistant");

    # Find the "Background of the Merger" section
    def findSection(self, file_path: str):
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
            assistant_id=self.__assistant.id
        )

        messages = list(self.__client.beta.threads.messages.list(
            thread_id=thread.id,
            run_id=run.id
        ));

        result = messages[0].content[0].text.value;

        # Delete the thread, file, & message but not the assistant (can reuse)
        self.__client.files.delete(msg_file.id);
        self.__client.beta.threads.delete(thread_id=thread.id);

        # Since we are using thread directly, the vector is created automatically.
        # API doesn't specify how to acquire a specific one, so I'm just flushing it out.
        vector_stores = self.__client.beta.vector_stores.list();
        for vector in vector_stores.data:
            self.__client.beta.vector_stores.delete(vector_store_id=str(vector.id));

        if run.status == "completed":
            return result;
        else:
            raise RuntimeError(f"Assistant failed with status: {run.status}");

    def deleteAssistant(self):
        self.__client.beta.assistants.delete(self.__assistant.id);
        print("Successfully deleted Assistant");