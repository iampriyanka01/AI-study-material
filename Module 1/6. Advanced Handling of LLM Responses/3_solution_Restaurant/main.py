import os
from dotenv import load_dotenv
from menuAssistant import MenuAssistant

load_dotenv()

def main():
    test_queries = [
        "What should I order at Sakura Sushi?",
        "What's the best dish at Bella Pasta?",
        "Tell me a joke",
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
