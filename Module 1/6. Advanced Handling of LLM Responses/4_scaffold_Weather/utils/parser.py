"""
WeatherParser module
Handles parsing, validation, and formatting of weather data from OpenWeatherMap API.
"""

class WeatherParser:
    def parse_weather_data(self, raw_data: dict) -> dict:
        return {
            "city": raw_data.get("name"),
            "country": raw_data.get("sys", {}).get("country", "N/A"),
            "temperature": raw_data.get("main", {}).get("temp"),
            "feels_like": raw_data.get("main", {}).get("feels_like"),
            "humidity": raw_data.get("main", {}).get("humidity"),
            "description": raw_data.get("weather", [{}])[0].get("description"),
            "wind_speed": raw_data.get("wind", {}).get("speed"),
        }

    def validate_weather_data(self, weather_data: dict) -> bool:
        required = ["city", "temperature", "description", "humidity", "wind_speed"]
        return all(weather_data.get(field) is not None for field in required)

    def format_weather_for_llm(self, weather_data: dict) -> str:
        return (
            f"City: {weather_data.get('city', 'Unknown')}\n"
            f"Country: {weather_data.get('country', 'N/A')}\n"
            f"Temperature: {weather_data.get('temperature', 'N/A')}°C\n"
            f"Feels Like: {weather_data.get('feels_like', 'N/A')}°C\n"
            f"Description: {weather_data.get('description', 'N/A')}\n"
            f"Humidity: {weather_data.get('humidity', 'N/A')}%\n"
            f"Wind Speed: {weather_data.get('wind_speed', 'N/A')} m/s\n"
        )

