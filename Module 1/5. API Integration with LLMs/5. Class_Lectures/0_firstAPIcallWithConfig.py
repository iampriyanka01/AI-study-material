import os #Loads Python’s built‑in os module, which lets you interact with environment variables and the operating system.
from google import genai #Imports the genai library from Google’s SDK. This is the client you’ll use to talk to Gemini models.
from dotenv import load_dotenv #Imports the load_dotenv function, which reads environment variables from a .env file
from google.genai.types import GenerateContentConfig


#This looks for a .env file in your project directory and loads any key‑value pairs inside it into your environment
load_dotenv() 

#Retrieves the value of the environment variable GOOGLE_API_KEY and stores it in the variable API_KEY.
API_KEY=os.getenv("GOOGLE_API_KEY") 

#Initializes a Gemini client using your API key so you can send requests to Google’s AI models.
client = genai.Client(api_key=API_KEY)

# Step 2: Define system instruction & generation config
config = GenerateContentConfig(
    system_instruction=[
        "As a tokenizer when you are given an input sentence, output a Python-style list of tokens, "
        "splitting words, subwords (e.g., contractions), and punctuation. "
    ],
    max_output_tokens=1000,
    temperature=0.0, # deterministic output
    top_p=1.0,
    top_k=1
)

#Calls the generate_content method on the Gemini model.
response = client.models.generate_content(
    model='gemini-3.6-flash',
    contents='What is the capital of France?',
    config=config
)

#Prints the text of the response from the model to the console.
print(f"Response:\n") 
print(response.text)
