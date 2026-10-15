from weatherAssistant import WeatherAssistant
from dotenv import load_dotenv

load_dotenv()

def main():
    """Main function to demonstrate the weather assistant functionality."""
    test_cities = [
        "London",
        "New York", 
        "Tokyo",
        "Paris",
        "Sydney"
    ]
    
    print("Weather Data Parser & API Assistant")
    print("=" * 50)
    
    assistant = WeatherAssistant()
    
    for city in test_cities:
        print(f"\nGetting weather insights for {city}...")
        try:
            result = assistant.get_weather_insights(city)
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Temperature: {result.get('temperature', 'N/A')}°C")
                print(f"Description: {result.get('description', 'N/A')}")
                print(f"Recommendation: {result.get('recommendation', 'N/A')}")
        except Exception as e:
            print(f"Error: {str(e)}")
        print("-" * 50)


if __name__ == "__main__":
    main()
