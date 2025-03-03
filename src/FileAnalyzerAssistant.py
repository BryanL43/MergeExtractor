from Assistant import Assistant

class FileAnalyzerAssistant(Assistant):
    def __init__(self, api_key: str, name: str, instructions: str, query: str, model: str, temp: float = 1.00, top_p: float = 1.00):
        self.__query = query;
        super().__init__(api_key, name, instructions, model, temp, top_p);

    def _createAssistant(self):
        """Creates a new assistant if one doesn't already exist."""
        if self._assistant_id:
            print(f"Assistant {self._name} already exists with ID: {self._assistant_id}");
            return;

        assistant = self._client.beta.assistants.create(
            name=self._name,
            instructions=self._instructions,
            model=self._model,
            tools=[{"type": "file_search"}],
            temperature=self._temp,
            top_p=self._top_p
        );

        self._assistant_id = assistant.id;

        # Check if assistant exists in JSON and update it
        existing_assistant = next((a for a in self._assistants_data if a["name"] == self._name), None);

        if existing_assistant:
            existing_assistant["id"] = self._assistant_id;
        else:
            self._assistants_data.append({"id": self._assistant_id, "name": self._name});

        self._saveAssistants();

        print(f"Successfully created Assistant: {self._name} (ID: {self._assistant_id})");

    def analyzeDocument(self, file_path: str):
        """Analyzes an attached document using the assistant."""
        msg_file = self._client.files.create(
            file=open(file_path, "rb"),
            purpose="assistants"
        );

        thread = self._client.beta.threads.create(
            messages = [
                {
                    "role": "user",
                    "content": self.__query,
                    "attachments": [
                        {
                            "file_id": msg_file.id,
                            "tools": [{"type": "file_search"}]
                        },
                    ]
                }
            ]
        );

        run = self._client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=self._assistant_id
        );

        messages = list(self._client.beta.threads.messages.list(
            thread_id=thread.id,
            run_id=run.id
        ));

        result = messages[0].content[0].text.value;

        # Delete the thread, file, & message but not the assistant (can reuse)
        self._client.files.delete(msg_file.id);
        self._client.beta.threads.delete(thread_id=thread.id);

        if run.status == "completed":
            return result;
        else:
            raise RuntimeError(f"Assistant failed with status: {run.status}");
