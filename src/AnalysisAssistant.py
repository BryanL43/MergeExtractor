import json
from Assistant import Assistant

CONFIG_FILE = "./config/ClassifierInstr.txt";

class AnalysisAssistant(Assistant):
    def __init__(
            self, 
            api_key: str, 
            name: str, 
            model: str, 
            temp: float = 1.00, 
            top_p: float = 1.00
        ):

        with open(CONFIG_FILE, "r") as file:
            instructions = file.read();
        
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
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "summarization_reporting",
                        "description": "Returns a structured summary including details about a merger deal initiation and its motivations.",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "required": [
                                "initiator",
                                "date_of_initiation",
                                "type_of_initiation",
                                "stated_reasons",
                                "key_figures"
                            ],
                            "properties": {
                                "initiator": {
                                    "type": "string",
                                    "description": "Who first proposed the merger."
                                },
                                "date_of_initiation": {
                                    "type": "string",
                                    "description": "When the proposal was made."
                                },
                                "type_of_initiation": {
                                    "type": "string",
                                    "description": "The type of initiation. Either an Acquirer-Initiated Deal, Target-Initiated Deal, or Third-Party-Initiated Deal."
                                },
                                "stated_reasons": {
                                    "type": "string",
                                    "description": "Why the merger was proposed."
                                },
                                "key_figures": {
                                    "type": "string",
                                    "description": "Key Figures Involved: Executives, stakeholders, or board members."
                                }
                            },
                            "additionalProperties": False
                        }
                    }
                }
            ],
            response_format={"type": "json_object"},
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

    def analyzeDocument(self, text: str):
        """Analyzes an attached document using the assistant."""
        thread = self._client.beta.threads.create(
            messages = [
                {
                    "role": "user",
                    "content": text
                }
            ]
        );

        run = self._client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=self._assistant_id
        );

        # Extract the json object from the function in a hacky manner.
        # This avoids openai messy APIs.
        result = None;
        for tool in run.required_action.submit_tool_outputs.tool_calls:
            if tool.function.name == "summarization_reporting":
                result = json.loads(tool.function.arguments);
                break;
        
        self._client.beta.threads.delete(thread_id=thread.id);

        return result;
