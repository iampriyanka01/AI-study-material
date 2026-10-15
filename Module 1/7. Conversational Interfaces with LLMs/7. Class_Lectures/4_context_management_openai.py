#This script loads your API key, connects to Google’s Gemini API using the OpenAI client, 
# sends a chat conversation, and prints the assistant’s response.

from openai import OpenAI # Imports the OpenAI client class, which lets you interact with APIs that follow the OpenAI‑style interface (here, Google’s Gemini API is exposed in that format).
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY=os.getenv("GOOGLE_API_KEY")

#Creates an OpenAI client object.
client = OpenAI(
    api_key=API_KEY,#Authenticates requests with your Google API key.    
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/" #OpenAPI part of Google’s Gemini API.
)

#This is a chat message list.
messages = [
    {"role": "system", 
     "content": "You are a career counselor. Give clear, empathetic advice."},
    {"role": "user", 
     "content": "Hi, I'm Nikhil. I enjoy math and programming and I just completed my B.Sc."}
]

#Send request to Gemini
response = client.chat.completions.create( #This comes from the OpenAI‑style client and calls the Chat Completions API.
    messages=messages,
    model='gemini-3.6-flash'
)

#Extract assistant reply
assistant_reply = response.choices[0].message.content

print("Assistant:", assistant_reply)

