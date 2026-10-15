import os
from dotenv import load_dotenv
from google import genai
from config import SYSTEM_PROMPT

# Load environment variables
load_dotenv()


class TechSupportChat:
    def __init__(self, api_key=None):
        # Initialize API key
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found")

        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)

        # Model name expected by tests
        self.model = "gemini-2.5-flash"

        # Conversation history storage
        self.conversation_history = []

        # Troubleshooting state storage
        self.troubleshooting_state = {
            "user_os": None,
            "device_type": None,
            "issue_type": None,
            "attempted_steps": [],
            "expertise_level": None,
            "attempted_solutions": []
        }

    def add_to_history(self, role, content, metadata=None):
        entry = {"role": role, "content": content}
        if metadata:
            entry["metadata"] = metadata
        self.conversation_history.append(entry)

        # Limit history length
        if len(self.conversation_history) > 20:
            self.conversation_history.pop(0)

    def update_troubleshooting_state(self, key, value):
        if key == "attempted_steps":
            if value not in self.troubleshooting_state["attempted_steps"]:
                self.troubleshooting_state["attempted_steps"].append(value)
        elif key == "attempted_solutions":
            if value not in self.troubleshooting_state["attempted_solutions"]:
                self.troubleshooting_state["attempted_solutions"].append(value)
        else:
            self.troubleshooting_state[key] = value

    def analyze_user_input(self, user_input):
        """Basic parser to update troubleshooting state from user input."""
        if "Windows 11" in user_input:
            self.update_troubleshooting_state("user_os", "Windows 11")
        if "Mac" in user_input:
            self.update_troubleshooting_state("user_os", "MacOS")
        if "WiFi" in user_input:
            self.update_troubleshooting_state("issue_type", "Connectivity")
        if "crash" in user_input.lower():
            self.update_troubleshooting_state("issue_type", "Software Crash")

    def get_conversation_context(self):
        context = SYSTEM_PROMPT + "\n\n"
        context += "TROUBLESHOOTING CONTEXT\n"
        for k, v in self.troubleshooting_state.items():
            context += f"- {k}: {v}\n"

        context += "\nConversation History:\n"
        for msg in self.conversation_history:
            context += f"{msg['role'].capitalize()}: {msg['content']}\n"

        return context

    def chat(self, user_input):
        self.add_to_history("user", user_input)
        self.analyze_user_input(user_input)

        context = self.get_conversation_context()

        response = self.client.models.generate_content(
            model=self.model,
            contents=context
        )

        reply = response.text
        self.add_to_history("assistant", reply)
        return f'Support: "{reply}"'

    def reset_conversation(self):
        self.conversation_history = []
        self.troubleshooting_state = {
            "user_os": None,
            "device_type": None,
            "issue_type": None,
            "attempted_steps": [],
            "expertise_level": None,
            "attempted_solutions": []
        }

    def get_troubleshooting_summary(self):
        summary = "Troubleshooting Summary:\n"
        for k, v in self.troubleshooting_state.items():
            summary += f"- {k}: {v}\n"
        return summary


def answer_support_query(query, assistant=None):
    try:
        if assistant is None:
            assistant = TechSupportChat()
        return assistant.chat(query)
    except Exception as e:
        return f"Error: {e}"


def main():
    print("Technical Support Assistant - Multi-Turn Chat")
    print("=" * 70)

    sample_conversations = [
        [
            "My browser keeps crashing every time I try to open it.",
            "I'm using Chrome on Windows 11.",
            "I restarted already, and it crashes right when I click the Chrome icon.",
            "Yes, it opens fine in safe mode."
        ],
        [
            "My WiFi keeps disconnecting randomly throughout the day.",
            "It's happening on my laptop, but my phone stays connected fine.",
            "It's a Dell Inspiron from 2021, and it seems to happen mostly when I'm on video calls.",
            "I updated the driver but the issue is still happening."
        ]
    ]

    assistant = TechSupportChat()

    for i, conversation in enumerate(sample_conversations):
        print(f"\n--- Sample Troubleshooting Session {i+1} ---")
        for query in conversation:
            print(f'User: "{query}"')
            response = answer_support_query(query, assistant)
            print(response)

        summary = assistant.get_troubleshooting_summary()
        print(f"\n{summary}")
        assistant.reset_conversation()


if __name__ == "__main__":
    main()
