import os
from geminiClient import GeminiClient
from dotenv import load_dotenv

# TODO: Load environment variables
load_dotenv()

def answer_student_query(query: str) -> str:
    """Answer a student's query about campus events."""
    try:            
    # TODO: Create llmClient instance
        client = GeminiClient()
    # TODO: Call the API
        response = client.call_api(query)
        return response
    # TODO: Handle exceptions and return error message if needed
    except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"


def main():
    """Main function to demonstrate the API calling functionality."""
    test_queries = [
        "When is the Tech Fest happening?",
        "Where is the Career Fair located?",
        "Tell me about the Hackathon",
        "What's the schedule for the Music Concert?",
        "Tell me a joke",  # Non-event query
        "What's the weather like?"  # Non-event query
    ]
    
    print("Campus Event Q&A Assistant")
    print("=" * 40)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = answer_student_query(query)
        print(f"Response: {result}")
        print("-" * 40)


if __name__ == "__main__":
    main()
