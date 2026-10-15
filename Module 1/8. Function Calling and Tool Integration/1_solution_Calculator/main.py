import os
from dotenv import load_dotenv
from google import genai
from utils.config import SYSTEM_PROMPT, add_numbers_mock, subtract_numbers_mock
from utils.parser import parse_function_call

load_dotenv()

class CalculatorAssistant:
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-3.6-flash"
    
    def handle_function_call(self, function_name, parameters):
        """Handle the actual function call execution"""
        try:
            if function_name == "add_numbers":
                num1 = float(parameters.get("num1", 0))
                num2 = float(parameters.get("num2", 0))
                result = add_numbers_mock(num1, num2)
                return f"The result of {num1} + {num2} is {result}"
            
            elif function_name == "subtract_numbers":
                num1 = float(parameters.get("num1", 0))
                num2 = float(parameters.get("num2", 0))
                result = subtract_numbers_mock(num1, num2)
                return f"The result of {num1} - {num2} is {result}"
            
            else:
                return f"Unknown function: {function_name}"
                
        except Exception as e:
            return f"Error executing function {function_name}: {str(e)}"
    
    
    def chat(self, user_input):
        """Main chat function that handles user input and decides whether to call functions"""
        try:
            # Prepare the conversation context
            conversation_context = f"{SYSTEM_PROMPT}\n\nUser: {user_input}\n\nAssistant:"
            
            # Call Gemini API
            response = self.client.models.generate_content(
                model=self.model,
                contents=conversation_context
            )
            
            if response.text is None or response.text.strip() == "":
                return "I didn't receive a valid response. Please try again."
            
            response_text = response.text.strip()
            
            # Check if response contains function call
            function_name, parameters = parse_function_call(response_text)
            
            if function_name and parameters is not None:
                # Execute the function call
                function_result = self.handle_function_call(function_name, parameters)
                return function_result
            else:
                # Return conversational response
                return response_text
                
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

def answer_math_query(query, assistant=None):
    """Convenient wrapper function for math queries"""
    try:
        if assistant is None:
            assistant = CalculatorAssistant()
        
        response = assistant.chat(query)
        return response
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def interactive_mode():
    """Interactive calculator mode"""
    print("Simple Calculator Assistant - Interactive Mode")
    print("Type 'quit' to exit")
    print("=" * 40)
    
    assistant = CalculatorAssistant()
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Thank you for using the calculator assistant. Goodbye!")
            break
        
        if not user_input:
            continue
        
        response = assistant.chat(user_input)
        print(f"\nCalculator: {response}")

def main():
    """Main function to demonstrate the calculator assistant."""
    print("Simple Calculator Assistant")
    print("=" * 40)
    
    # Sample test queries
    sample_queries = [
        "What is 5 plus 3?",
        "Calculate 10 minus 4",
        "Add 15 and 25",
        "Hello, how are you?",
        "What can you do?"
    ]
    
    assistant = CalculatorAssistant()
    
    print("\n--- Sample Queries ---")
    for query in sample_queries:
        print(f"\nUser: {query}")
        response = answer_math_query(query, assistant)
        print(f"Calculator: {response}")
    
    print("\n" + "-" * 40)
    
    # Offer interactive mode
    print("\nWould you like to try interactive mode? (y/n)")
    if input().lower().startswith('y'):
        interactive_mode()

if __name__ == "__main__":
    main()