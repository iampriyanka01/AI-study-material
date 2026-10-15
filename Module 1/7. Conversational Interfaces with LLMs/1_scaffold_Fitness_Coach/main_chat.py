import os
from dotenv import load_dotenv
from google import genai
from utils.config import SYSTEM_PROMPT   # Import system prompt

# Load environment variables
load_dotenv()

class ChatAssistant:
    def __init__(self, api_key=None):
        # Initialize API key from parameter or env var GEMINI_API_KEY
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY in .env or pass explicitly.")

        # ✅ Initialize Gemini client first
        self.client = genai.Client(api_key=self.api_key)

        # ✅ Create a chat session
        self.chat_session = self.client.chats.create(model="gemini-3.6-flash")

        # ✅ Send system prompt as the first message
        self.chat_session.send_message(SYSTEM_PROMPT)

    def send(self, user_input):
        """Send user input to Gemini chat and return assistant response."""
        try:
            response = self.chat_session.send_message(user_input)
            return response.text.strip()
        except Exception as e:
            return f"Error during chat: {str(e)}"

    def reset_conversation(self):
        """Clear conversation history by creating a new chat session and re-sending system prompt."""
        self.chat_session = self.client.chats.create(model="gemini-3.6-flash")
        self.chat_session.send_message(SYSTEM_PROMPT)


def answer_fitness_query(query, assistant=None):
    """Helper to handle fitness queries with error management."""
    try:
        if assistant is None:
            assistant = ChatAssistant()
        return assistant.send(query)   # ✅ use .send() method
    except Exception as e:
        return f"Error answering query: {str(e)}"


def main():
    """Main function to demonstrate the fitness coach assistant."""
    print("Personal Fitness Coach - Multi-Turn Chat")
    print("Type 'quit' to exit, 'reset' to start new conversation")
    print("=" * 50)
    
    sample_conversations = [
        [
            "I'm new to working out and want to build muscle. Where should I start?",
            "I can work out 3 days a week.",
            "I can't do pull-ups yet, what alternatives do you suggest?"
        ],
        [
            "I want to lose weight but don't know what to eat.",
            "I eat out a lot and drink soda daily.",
            "I like Italian and Mexican food."
        ]
    ]
    
    assistant = ChatAssistant()
    
    for i, conversation in enumerate(sample_conversations):
        print(f"\n--- Sample Conversation {i+1} ---")
        for query in conversation:
            print(f"\nUser: {query}")
            response = answer_fitness_query(query, assistant)
            print(f"Coach: {response}")
        assistant.reset_conversation()

if __name__ == "__main__":
    main()
