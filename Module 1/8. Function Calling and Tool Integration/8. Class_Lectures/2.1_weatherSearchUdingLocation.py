from google import genai
from google.genai import types
import requests
from dotenv import load_dotenv
import os

load_dotenv()


GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

def get_current_temperature(location):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"

    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        print(f"The current temperature in {location} is {data['main']['temp']}°C.")
        return data['main']['temp']
    else:
        raise Exception(f"Error fetching weather: {data.get('message', 'Unknown error')}")

get_current_temperature("Rome")