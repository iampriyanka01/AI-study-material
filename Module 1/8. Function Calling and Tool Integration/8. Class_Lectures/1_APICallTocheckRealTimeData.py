import pprint
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()


GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Initialize the client with your API key
genai.configure(api_key=GEMINI_API_KEY)

# Create a model instance
model = genai.GenerativeModel("gemini-3.6-flash")

# Send a query to the model
response = model.generate_content("Who is the current chief minister of Tamil Nadu?")

# Print the response text
print(response.text)

# Pretty-print the response object
pprint.pprint(response.text)

