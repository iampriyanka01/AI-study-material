import os #
from dotenv import load_dotenv 
from google import genai 

#This looks for a .env file in your project directory and loads any key‑value pairs inside it into your environment
load_dotenv() 

#Retrieves the value of the environment variable GOOGLE_API_KEY and stores it in the variable API_KEY.
API_KEY=os.getenv("GOOGLE_API_KEY") 

#Initializes a Gemini client using your API key so you can send requests to Google’s AI models.
client = genai.Client(api_key=API_KEY)

#Calls the generate_content method on the Gemini model.
response = client.models.generate_content(
    model='gemini-3.6-flash',
    contents='What is the capital of France?'
)

#Prints the text of the response from the model to the console.
print(f"Response: {response.text}") 