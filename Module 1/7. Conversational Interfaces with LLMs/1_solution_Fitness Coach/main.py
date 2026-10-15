import os
from dotenv import load_dotenv
from google import genai
from utils.config import SYSTEM_PROMPT

load_dotenv()

class ChatAssistant:
    
    def __init__(self, api_key=None, max_history=10):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-3.6-flash"
        self.conversation_history = []
        self.max_history = max_history
    
    def add_to_history(self, role, content):
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # Keep only the last max_history messages (excluding system prompt)
        if len(self.conversation_history) > self.max_history:
            # Keep system message if it exists, remove oldest user/assistant messages
            system_messages = [msg for msg in self.conversation_history if msg["role"] == "system"]
            other_messages = [msg for msg in self.conversation_history if msg["role"] != "system"]
            
            # Keep only the most recent messages
            recent_messages = other_messages[-(self.max_history - len(system_messages)):]
            self.conversation_history = system_messages + recent_messages
    
    def get_conversation_context(self):
        """Format conversation history for API call"""
        if not self.conversation_history:
            return SYSTEM_PROMPT
        
        # Build conversation context
        context_parts = [SYSTEM_PROMPT]
        
        for message in self.conversation_history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                context_parts.append(f"User: {content}")
            elif role == "assistant":
                context_parts.append(f"Coach: {content}")
        
        return "\n\n".join(context_parts)
    
    def chat(self, user_input):
        """Main chat function with conversation history"""
        try:
            # Add user input to history
            self.add_to_history("user", user_input)
            
            # Get full conversation context
            conversation_context = self.get_conversation_context()
            
            # Add current user input to context
            full_prompt = f"{conversation_context}\n\nUser: {user_input}\n\nCoach:"
            
            # Call Gemini API
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt
            )
            
            if response.text is None or response.text.strip() == "":
                return "I didn't receive a valid response. Please try again."
            
            response_text = response.text.strip()
            
            # Add assistant response to history
            self.add_to_history("assistant", response_text)
            
            return response_text
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []

def answer_fitness_query(query, assistant=None):
    """Convenient wrapper function"""
    try:
        if assistant is None:
            assistant = ChatAssistant()
        
        response = assistant.chat(query)
        return response
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def interactive_mode():
    """Interactive chat mode"""
    print("Interactive Mode")
    print("Type 'quit' to exit, 'reset' to start new conversation")
    print("=" * 50)
    
    assistant = ChatAssistant()
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Stay healthy! Goodbye!")
            break
        
        if user_input.lower() == 'reset':
            assistant.reset_conversation()
            print("Conversation reset. Starting fresh!")
            continue
        
        if not user_input:
            continue
        
        response = assistant.chat(user_input)
        print(f"\nCoach: {response}")

def main():
    """Main function to demonstrate the fitness coach assistant."""
    print("Chat Completion")
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
    
    # Demonstrate conversation examples
    assistant = ChatAssistant()
    
    for i, conversation in enumerate(sample_conversations):
        print(f"\n--- Sample Conversation {i+1} ---")
        for query in conversation:
            print(f"\nUser: {query}")
            response = answer_fitness_query(query, assistant)
            print(f"Coach: {response}")
        
        assistant.reset_conversation()
        print("\n" + "-" * 40)
    
    # Offer interactive mode
    print("\nWould you like to try interactive mode? (y/n)")
    if input().lower().startswith('y'):
        interactive_mode()

if __name__ == "__main__":
    main()
