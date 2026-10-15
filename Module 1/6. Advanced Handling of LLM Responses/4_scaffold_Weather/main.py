import os
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from utils.config import SYSTEM_PROMPT
from utils.parser import WeatherParser
from google import genai
import json

# Load environment variables
load_dotenv()


class WeatherAPIClient:
    """Client for fetching weather data from OpenWeatherMap API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("WEATHER_API_KEY")
        if not self.api_key:
            raise ValueError("WEATHER_API_KEY not found in environment variables.")
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"        

    def get_weather(self, city: str) -> Dict[str, Any]:
        try:
            resp = requests.get(
                self.base_url,
                params={"q": city, 
                        "appid": self.api_key, 
                        "units": "metric"},
                timeout=10
            )

            # Handle specific status codes explicitly
            if resp.status_code == 401:
                raise ValueError("Invalid API key")
            elif resp.status_code == 404:
                raise ValueError(f"City '{city}' not found")
            elif resp.status_code == 429:
                raise ValueError("API rate limit exceeded")

            # For other errors, raise_for_status will trigger HTTPError
            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout:
            raise ValueError("Request timeout - please try again.")
        except requests.exceptions.ConnectionError:
            raise ValueError("Network error - please check your connection.")
        except requests.exceptions.HTTPError as e:
            # Catch any other HTTP errors (e.g., 500 server error)
            if e.response is not None:
                raise ValueError(f"HTTP error: {e.response.status_code} {e.response.reason}")
            else:
                raise ValueError("HTTP error occurred but no response available")


class LLMClient:
    """Client for calling Gemini API to generate weather insights."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found.")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self.client = genai.Client(api_key=self.api_key)

    def generate_insights(self, weather_data: Dict[str, Any]) -> str:
        parser = WeatherParser()
        formatted = parser.format_weather_for_llm(weather_data)
        prompt = SYSTEM_PROMPT.format(weather_data=formatted)
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            if not response.text or response.text.strip() == "":
                return "No insights available."
            return json.loads(response.text)
        except Exception as e:
            return f"LLM error: {str(e)}"


class WeatherAssistant:
    """Main assistant that coordinates weather data fetching and insights generation."""

    def __init__(self):
        self.weather_client = WeatherAPIClient()
        self.llm_client = LLMClient()
        self.parser = WeatherParser()

    def get_weather_insights(self, city: str) -> Dict[str, Any]:
        try:
            raw = self.weather_client.get_weather(city)
            parsed = self.parser.parse_weather_data(raw)
            if not self.parser.validate_weather_data(parsed):
                raise ValueError("Invalid weather data received")

            llm_output = self.llm_client.generate_insights(parsed)

            return {
                **parsed,
                "recommendation": llm_output["recommendation"],
                "insights": llm_output["insights"]
}
        except Exception as e:
            return {
                "error": str(e),
                "city": city,
                "recommendation": "Unable to get weather data..."
            }


def main():
    test_cities = ["London", 
                   "New York", 
                   #"Tokyo", 
                   #"Paris", 
                   #"Sydney"
                   ]

    print("Weather Data Parser & API Assistant")
    print("=" * 50)

    assistant = WeatherAssistant()

    for city in test_cities:
        #print(f"\nGetting weather insights for {city}...")
        result = assistant.get_weather_insights(city)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
             print(f"\n")
             print(json.dumps(result, indent=2))
        #print("-" * 50)


if __name__ == "__main__":
    main()
