from openai import OpenAI
import json
import os
import sys

class Assistant:
    def __init__(self, api_key: str, name: str, instructions: str, model: str, temp: float = 1.00, top_p: float = 1.00):
        self._client = OpenAI(api_key=api_key);
        self._name = name;
        self._instructions = instructions;
        self._model = model;
        self._temp = temp;
        self._top_p = top_p;

        self._assistant_id = None;

        # Load existing assistants from JSON
        self._assistants_data = self._loadAssistants();

        # Try to find an existing assistant
        existing_assistant = next((a for a in self._assistants_data if a["name"] == self._name), None);

        if existing_assistant:
            self._assistant_id = existing_assistant["id"];

            # Validate assistant existence
            try:
                self._client.beta.assistants.retrieve(self._assistant_id);
                print(f"Using existing Assistant: {self._name} (ID: {self._assistant_id})");
                return;
            except Exception:
                print(f"Assistant {self._name} found in JSON but does not exist. Creating a new one...");
                self._assistant_id = None;  # Reset to force creation

        # If assistant was not found or was invalid, create a new one
        self._createAssistant();

    def _createAssistant(self):
        """To be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement _createAssistant");

    def _loadAssistants(self):
        """Load assistants from JSON file"""
        if os.path.exists("assistantData.json"):
            try:
                with open("assistantData.json", "r") as json_file:
                    return json.load(json_file) or [];
            except json.JSONDecodeError:
                return [];
        return [];

    def _saveAssistants(self):
        """Save assistant data back to JSON"""
        with open("assistantData.json", "w") as json_file:
            json.dump(self._assistants_data, json_file, indent=4);

    def deleteAssistant(self):
        """Deletes the assistant and updates JSON file"""
        if self._assistant_id:
            self._client.beta.assistants.delete(self._assistant_id);

        self._assistants_data = [a for a in self._assistants_data if a["id"] != self._assistant_id];

        if self._assistants_data:
            self._saveAssistants();
        else:
            os.remove("assistantData.json");

        print(f"Successfully deleted Assistant: {self._name}");
