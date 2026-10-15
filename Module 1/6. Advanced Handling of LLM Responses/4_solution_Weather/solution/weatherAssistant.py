from solution.utils.weatherparser import WeatherParser
from weatherAPIClient import WeatherAPIClient
from geminiClient import GeminiClientClient
from typing import Optional, Dict, Any

class WeatherAssistant:
    """Main assistant that coordinates weather data fetching and insights generation."""
    
    def __init__(self):
        self.weatherclient = WeatherAPIClient()
        self.geminiClient = GeminiClientClient()
        self.weatherparser = WeatherParser()
    
    def get_weather_insights(self, city: str) -> Dict[str, Any]:
        """Get weather data and generate insights for a given city."""
        try:
            # Fetch  parsed weather data from city
            raw_weather = self.weatherclient.get_weather(city)
            parsed_weather = self.weatherparser.parse_weather_data(raw_weather)  
            weather_validation_flag = self.weatherparser.validate_weather_data(parsed_weather)          
            if not weather_validation_flag:
                raise ValueError("Invalid weather data received")
            
            # Generate insights using LLM using Parsed weather data and Return structured response
            insights = self.geminiClient.generate_insights(parsed_weather)
            return {
                "city": parsed_weather.get("city", city),
                "country": parsed_weather.get("country", "Unknown"),
                "temperature": parsed_weather.get("temperature"),
                "feels_like": parsed_weather.get("feels_like"),
                "description": parsed_weather.get("description"),
                "humidity": parsed_weather.get("humidity"),
                "wind_speed": parsed_weather.get("wind_speed"),
                "recommendation": insights,
                "insights": insights
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "city": city,
                "recommendation": f"Unable to get weather data for {city}: {str(e)}"
            }