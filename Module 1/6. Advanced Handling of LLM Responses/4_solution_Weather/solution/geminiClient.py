
from google import genai
from typing import Optional, Dict, Any
from utils.config import SYSTEM_PROMPT
import os
from dotenv import load_dotenv


class GeminiClient:
    """Client for calling Gemini API to generate weather insights."""
    
    def __init__(self, api_key: Optional[str] = None):

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        self.client = genai.Client(api_key=self.api_key)
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    def generate_insights(self, weather_data: Dict[str, Any]) -> str:

        """Generate weather insights and recommendations using LLM."""
        try:
            formatted_data = self.format_weather_for_llm(weather_data)
            prompt = SYSTEM_PROMPT.format(weather_data=formatted_data)
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            if response.text is None or response.text.strip() == "":
                return "Unable to generate weather insights at this time."
            
            return response.text.strip()
            
        except Exception as e:
            return f"Error generating insights: {str(e)}"
    
    def format_weather_for_llm(self, weather_data: Dict[str, Any]) -> str:
        """Format weather data for LLM prompt."""
        return f"""
                City: {weather_data.get('city', 'Unknown')}
                Country: {weather_data.get('country', 'Unknown')}
                Temperature: {weather_data.get('temperature', 'N/A')}°C
                Feels Like: {weather_data.get('feels_like', 'N/A')}°C
                Description: {weather_data.get('description', 'N/A')}
                Humidity: {weather_data.get('humidity', 'N/A')}%
                Wind Speed: {weather_data.get('wind_speed', 'N/A')} m/s
                """