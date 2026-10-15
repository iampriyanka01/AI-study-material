"""
Weather data parser module.
Handles parsing and validation of weather API responses.
"""

from typing import Dict, Any


class WeatherParser:
    """Parser for weather API responses."""
    
    def parse_weather_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw weather API response into structured format."""

        # Extract city information
        city = raw_data.get('name', 'Unknown')
        country = raw_data.get('sys', {}).get('country', 'Unknown')
        
        # Extract main weather data
        main_data = raw_data.get('main', {})
        temperature = main_data.get('temp')
        feels_like = main_data.get('feels_like')
        humidity = main_data.get('humidity')
        
        # Extract weather description
        weather_list = raw_data.get('weather', [])
        description = weather_list[0].get('description', 'Unknown') if weather_list else 'Unknown'
        '''
        # Check if weather_list is not empty
        if weather_list:
            description = weather_list[0].get('description', 'Unknown')
        else:
            description = 'Unknown'
        '''
        # Extract wind data
        wind_speed = raw_data.get('wind', {}).get('speed')
        
        return {
            'city': city,
            'country': country,
            'temperature': temperature,
            'feels_like': feels_like,
            'description': description,
            'humidity': humidity,
            'wind_speed': wind_speed
        }
    
    def validate_weather_data(self, weather_data: Dict[str, Any]) -> bool:
        """Validate that weather data has required fields."""
        required_fields = ['city', 'temperature', 'description', 'humidity', 'wind_speed']
        return all(field in weather_data and weather_data[field] is not None for field in required_fields)
        
        '''
        result = True
        for field in required_fields:
            if field not in weather_data:
                result = False
                break            
            if weather_data[field] is None:
                result = False
                break
        return(result)  
        '''
    

    def format_weather_for_llm(self, weather_data: Dict[str, Any]) -> str:
        """Format weather data for LLM prompt."""
        return f"""City: {weather_data.get('city', 'Unknown')}
                    Country: {weather_data.get('country', 'Unknown')}
                    Temperature: {weather_data.get('temperature', 'N/A')}°C
                    Feels Like: {weather_data.get('feels_like', 'N/A')}°C
                    Description: {weather_data.get('description', 'N/A')}
                    Humidity: {weather_data.get('humidity', 'N/A')}%
                    Wind Speed: {weather_data.get('wind_speed', 'N/A')} m/s"""
