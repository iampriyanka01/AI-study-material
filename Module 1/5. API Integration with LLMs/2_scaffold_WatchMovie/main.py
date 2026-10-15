import os
from geminiClient import GeminiClient
from dotenv import load_dotenv

# TODO: Load environment variables
load_dotenv()

def answer_student_query(query):
    """Answer a User's query about Movie."""
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
    """Main function to demonstrate the movie recommendation assistant."""
    print("Movie Recommendation Assistant")
    print("=" * 50)
    
    sample_queries = [
        "I want to watch a good action movie tonight",
        "Recommend me something funny",
        "What's a good sci-fi movie?",
        "I'm in the mood for horror",
        "What's the weather like today?"
    ]
    
    for query in sample_queries:
        print(f"\nQuery: {query}")
        response = answer_student_query(query)
        print(f"Response: {response}")
        print("-" * 40)

if __name__ == "__main__":
    main()
