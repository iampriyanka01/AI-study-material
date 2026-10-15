from google import genai
from google.genai import types
import requests
import datetime
from dotenv import load_dotenv
import os

load_dotenv()


GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

def get_temperature_for_date(location: str, date_str: str):
    """
    Fetch forecasted temperature for a given location and date.
    date_str format: 'YYYY-MM-DD HH:MM:SS'
    """
    # Convert date string to datetime
    target_dt = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

    # Forecast endpoint (5-day / 3-hour intervals)
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        raise Exception(f"Error fetching forecast: {data.get('message', 'Unknown error')}")

    # Find forecast entry closest to target date/time
    closest = min(
        data["list"],
        key=lambda entry: abs(
            datetime.datetime.fromtimestamp(entry["dt"]) - target_dt
        )
    )

    forecast_time = datetime.datetime.fromtimestamp(closest["dt"])
    temp = closest["main"]["temp"]

    print(f"Forecast for {location} at {forecast_time}: {temp}°C")
    return temp

# Example usage
get_temperature_for_date("Rome", "2026-08-14 15:00:00")
