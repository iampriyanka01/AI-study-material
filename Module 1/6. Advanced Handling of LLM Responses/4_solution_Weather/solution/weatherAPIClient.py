import requests
from typing import Optional, Dict, Any
import os

"""Client for fetching weather data from OpenWeatherMap API."""
class WeatherAPIClient:   
    
    def __init__(self, api_key: Optional[str] = None):

        self.api_key = api_key or os.getenv("WEATHER_API_KEY")
        if not self.api_key:
            raise ValueError("WEATHER_API_KEY not found in environment variables.")
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    def get_weather(self, city: str) -> Dict[str, Any]:
        """Fetch weather data for a given city."""
        try:
            params = {
                "q": city,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = requests.get(self.base_url, 
                                    params=params, 
                                    timeout=10)
            response.raise_for_status()            
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError("Invalid API key")
            elif e.response.status_code == 404:
                raise ValueError(f"City '{city}' not found")
            elif e.response.status_code == 429:
                raise ValueError("API rate limit exceeded")
            else:
                raise ValueError(f"API error: {e.response.status_code}")
        except requests.exceptions.Timeout:
            raise ValueError("Request timeout - please try again")
        except requests.exceptions.ConnectionError:
            raise ValueError("Network error - please check your connection")
        except Exception as e:
            raise ValueError(f"Unexpected error: {str(e)}")
