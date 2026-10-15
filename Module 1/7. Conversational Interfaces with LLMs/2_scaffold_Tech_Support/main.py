# main.py

import os
from dotenv import load_dotenv
from google import genai

from utils.config import SYSTEM_PROMPT
from utils.parser import parse_function_call

load_dotenv()

class CalculatorAssistant:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY in .env")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-3.5-flash"   # ✅ use Gemini 3.5

    def handle_function_call(self, function_name, parameters):
        if function_name == "add_numbers":
            result = parameters["num1"] + parameters["num2"]
            return f"The result of addition is {result}"
        elif function_name == "subtract_numbers":
            result = parameters["num1"] - parameters["num2"]
            return f"The result of subtraction is {result}"
        else:
            return "Sorry, I don’t recognize that function."

    def chat(self, user_input):
        try:
            # ✅ contents must be string(s), not dicts
            response = self.client.models.generate_content(
                model=self.model,
                contents=[f"{SYSTEM_PROMPT}\n\nUser: {user_input}"]
            )

            response_text = response.text.strip()
            function_name, params = parse_function_call(response_text)

            if function_name:
                return self.handle_function_call(function_name, params)
            else:
                return response_text

        except Exception as e:
            return f"Error: {str(e)}"


def answer_math_query(query, assistant=None):
    try:
        if assistant is None:
            assistant = CalculatorAssistant()
        return assistant.chat(query)
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    print("Simple Calculator Assistant (Gemini 3.5)")
    print("Type 'quit' to exit")
    print("=" * 40)

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


if __name__ == "__main__":
    main()
