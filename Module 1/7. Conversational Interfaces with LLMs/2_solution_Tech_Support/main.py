import os
from dotenv import load_dotenv
from google import genai
from utils.config import SYSTEM_PROMPT

load_dotenv()

class TechSupportChat:
    
    def __init__(self, api_key=None, max_history=15):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.5-flash"
        self.conversation_history = []
        self.max_history = max_history
        
        # Troubleshooting state tracking
        self.troubleshooting_state = {
            "user_os": None,
            "user_device": None,
            "issue_type": None,
            "attempted_solutions": [],
            "user_tech_level": "unknown",  # beginner, intermediate, advanced
            "system_info": {},
            "current_step": None,
            "escalation_needed": False
        }
    
    def add_to_history(self, role, content, metadata=None):
        """Add message to conversation history with optional troubleshooting metadata"""
        message = {
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        self.conversation_history.append(message)
        
        # Keep only the last max_history messages (excluding system prompt)
        if len(self.conversation_history) > self.max_history:
            system_messages = [msg for msg in self.conversation_history if msg["role"] == "system"]
            other_messages = [msg for msg in self.conversation_history if msg["role"] != "system"]
            
            recent_messages = other_messages[-(self.max_history - len(system_messages)):]
            self.conversation_history = system_messages + recent_messages
    
    def update_troubleshooting_state(self, key, value):
        """Update troubleshooting context"""
        if key in self.troubleshooting_state:
            if key == "attempted_solutions" and isinstance(value, str):
                if value not in self.troubleshooting_state[key]:
                    self.troubleshooting_state[key].append(value)
            else:
                self.troubleshooting_state[key] = value
    
    def get_conversation_context(self):
        """Format conversation history and troubleshooting state for API call"""
        context_parts = [SYSTEM_PROMPT]
        
        # Add troubleshooting state if available
        if any(v for v in self.troubleshooting_state.values() if v):
            state_info = []
            if self.troubleshooting_state["user_os"]:
                state_info.append(f"User OS: {self.troubleshooting_state['user_os']}")
            if self.troubleshooting_state["user_device"]:
                state_info.append(f"Device: {self.troubleshooting_state['user_device']}")
            if self.troubleshooting_state["issue_type"]:
                state_info.append(f"Issue Type: {self.troubleshooting_state['issue_type']}")
            if self.troubleshooting_state["attempted_solutions"]:
                state_info.append(f"Attempted Solutions: {', '.join(self.troubleshooting_state['attempted_solutions'])}")
            if self.troubleshooting_state["user_tech_level"] != "unknown":
                state_info.append(f"User Tech Level: {self.troubleshooting_state['user_tech_level']}")
            
            if state_info:
                context_parts.append(f"TROUBLESHOOTING CONTEXT: {'; '.join(state_info)}")
        
        # Add conversation history
        for message in self.conversation_history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                context_parts.append(f"User: {content}")
            elif role == "assistant":
                context_parts.append(f"Support: {content}")
        
        return "\n\n".join(context_parts)
    
    def analyze_user_input(self, user_input):
        """Analyze user input for troubleshooting information"""
        input_lower = user_input.lower()
        
        # Detect OS
        if "windows" in input_lower:
            self.update_troubleshooting_state("user_os", "Windows")
        elif "mac" in input_lower or "macos" in input_lower:
            self.update_troubleshooting_state("user_os", "macOS")
        elif "linux" in input_lower:
            self.update_troubleshooting_state("user_os", "Linux")
        
        # Detect device type
        if "laptop" in input_lower:
            self.update_troubleshooting_state("user_device", "Laptop")
        elif "desktop" in input_lower:
            self.update_troubleshooting_state("user_device", "Desktop")
        elif "phone" in input_lower:
            self.update_troubleshooting_state("user_device", "Phone")
        
        # Detect issue types
        if any(word in input_lower for word in ["crash", "crashes", "crashing", "freeze", "freezing"]):
            self.update_troubleshooting_state("issue_type", "Software Crash")
        elif any(word in input_lower for word in ["wifi", "internet", "connection", "network"]):
            self.update_troubleshooting_state("issue_type", "Connectivity")
        elif any(word in input_lower for word in ["login", "password", "account", "sign in"]):
            self.update_troubleshooting_state("issue_type", "Authentication")
        elif any(word in input_lower for word in ["slow", "performance", "lag", "sluggish"]):
            self.update_troubleshooting_state("issue_type", "Performance")
        
        # Detect attempted solutions
        if "restart" in input_lower or "reboot" in input_lower:
            self.update_troubleshooting_state("attempted_solutions", "System restart")
        if "update" in input_lower and "driver" in input_lower:
            self.update_troubleshooting_state("attempted_solutions", "Driver update")
        if "reinstall" in input_lower:
            self.update_troubleshooting_state("attempted_solutions", "Software reinstall")
    
    def chat(self, user_input):
        """Main chat function with troubleshooting context"""
        try:
            # Analyze user input for troubleshooting information
            self.analyze_user_input(user_input)
            
            # Add user input to history
            self.add_to_history("user", user_input)
            
            # Get full conversation context
            conversation_context = self.get_conversation_context()
            
            # Add current user input to context
            full_prompt = f"{conversation_context}\n\nUser: {user_input}\n\nSupport:"
            
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
        """Clear conversation history and reset troubleshooting state"""
        self.conversation_history = []
        self.troubleshooting_state = {
            "user_os": None,
            "user_device": None,
            "issue_type": None,
            "attempted_solutions": [],
            "user_tech_level": "unknown",
            "system_info": {},
            "current_step": None,
            "escalation_needed": False
        }
    
    def get_troubleshooting_summary(self):
        """Generate summary of troubleshooting session"""
        if not any(v for v in self.troubleshooting_state.values() if v):
            return "No troubleshooting information gathered yet."
        
        summary_parts = []
        
        if self.troubleshooting_state["issue_type"]:
            summary_parts.append(f"Issue: {self.troubleshooting_state['issue_type']}")
        
        if self.troubleshooting_state["user_os"] or self.troubleshooting_state["user_device"]:
            system_info = []
            if self.troubleshooting_state["user_os"]:
                system_info.append(self.troubleshooting_state["user_os"])
            if self.troubleshooting_state["user_device"]:
                system_info.append(self.troubleshooting_state["user_device"])
            summary_parts.append(f"System: {' '.join(system_info)}")
        
        if self.troubleshooting_state["attempted_solutions"]:
            solutions = ", ".join(self.troubleshooting_state["attempted_solutions"])
            summary_parts.append(f"Attempted: {solutions}")
        
        return " | ".join(summary_parts) if summary_parts else "No troubleshooting information available."

def answer_support_query(query, assistant=None):
    """Convenient wrapper function"""
    try:
        if assistant is None:
            assistant = TechSupportChat()
        
        response = assistant.chat(query)
        return response
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def interactive_mode():
    """Interactive technical support mode"""
    print("Technical Support Assistant - Interactive Mode")
    print("Type 'quit' to exit, 'reset' to start new conversation, 'summary' for troubleshooting summary")
    print("=" * 70)
    
    assistant = TechSupportChat()
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Thank you for using technical support. Have a great day!")
            break
        
        if user_input.lower() == 'reset':
            assistant.reset_conversation()
            print("Conversation reset. Starting fresh troubleshooting session!")
            continue
        
        if user_input.lower() == 'summary':
            summary = assistant.get_troubleshooting_summary()
            print(f"\nTroubleshooting Summary: {summary}")
            continue
        
        if not user_input:
            continue
        
        response = assistant.chat(user_input)
        print(f"\nSupport: {response}")

def main():
    """Main function to demonstrate the technical support assistant."""
    print("Chat Completion")
    print("=" * 50)
    
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
    
    # Demonstrate conversation examples
    assistant = TechSupportChat()
    
    for i, conversation in enumerate(sample_conversations):
        print(f"\n--- Sample Troubleshooting Session {i+1} ---")
        for query in conversation:
            print(f"\nUser: {query}")
            response = answer_support_query(query, assistant)
            print(f"Support: {response}")
        
        summary = assistant.get_troubleshooting_summary()
        print(f"\nTroubleshooting Summary: {summary}")
        assistant.reset_conversation()
        print("\n" + "-" * 50)
    
    # Offer interactive mode
    print("\nWould you like to try interactive mode? (y/n)")
    if input().lower().startswith('y'):
        interactive_mode()

if __name__ == "__main__":
    main()
