## 1. Required Imports
from google import genai
from google.genai import types
import requests
from dotenv import load_dotenv
import os

load_dotenv()


# Replace with your own keys or use environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print("Gemini key:", GEMINI_API_KEY)
OPENWEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
print("Weather key:", OPENWEATHER_API_KEY)

# ——— Get input from user ———

location1 = input("Enter location: ")

## 3. Function to be called
def get_current_temperature(location):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"

    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        #print(f"The current temperature in {location} is {data['main']['temp']}°C.")
        return data['main']['temp']
    else:
        raise Exception(f"Error fetching weather: {data.get('message', 'Unknown error')}")

#get_current_temperature(location1)

## 4. Function Schema

weather_function = {
    "name" : "get_current_temperature",
    "description": "Fetches the forecast weather for a given city with optional date. Use when user asks about weather, temperature, rain conditions, humidity, whther to take an umbrella or not",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "city": {
                "type": "STRING",
                "description": "Name of the city for which the weather forecast is required"
            },
            "date": {
                "type": "STRING",
                "description": "Date for which the weather forecast is required. Format should be YYYY-MM-DD(optional; default: today)"
            }
        },
        "required": ["city"]
    }
}

client = genai.Client(api_key=GEMINI_API_KEY)
tools = types.Tool(function_declarations=[weather_function])
config = types.GenerateContentConfig(tools=[tools])

user_prompt = f"What is the weather in {location1}?"

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=user_prompt,
    config=config
)

first_part = response.candidates[0].content.parts[0]

if first_part.function_call.name == "get_current_temperature":

    temperature = get_current_temperature(first_part.function_call.args.get("city"))
    print(f"LLM: Current temperature in {location1} is  {temperature}°C")
 
    