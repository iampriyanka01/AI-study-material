from dotenv import load_dotenv
from menuAssistant import MenuAssistant

# TODO: Load environment variables
load_dotenv()  # this loads variables from .env into os.environ

def main():
    """Main function to demonstrate the travel advisor functionality."""
    test_queries = [
        #"I want to visit Paris next week",
        "What's the best dish at Bella Pasta?",
        #"Planning a trip to Tokyo in December",
        #"What's the weather like in New York?",
        #"Should I go to Sydney this summer?",
        #"Tell me about London's weather",
        #"What's the best time to travel?",  # Non-travel query
        "How are you doing?"  # Non-travel query
    ]
    
    print("Restaurant Menu Assistant")
    print("=" * 40)
    
    assistant = MenuAssistant()
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = assistant.answer(query)
        print(f"Response: {result}")
        print("-" * 40)


if __name__ == "__main__":
    main()
