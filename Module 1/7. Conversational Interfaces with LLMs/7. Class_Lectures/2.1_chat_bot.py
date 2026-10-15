from google import genai 
from dotenv import load_dotenv
import os
 # Initialize Gemini client 

load_dotenv()


# Initialize a model client (example: Gemini 1.5 Pro)
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
               
 # Create a new chat session using Gemini's chat API 
chat = client.chats.create(model="gemini-3.6-flash") 

#input1 = input("Enter your message 1: ")
input1 = "who is the director of movie inception?"
 # First message — sets initial context 
response1 = chat.send_message(input1) 
print(response1.text)

#input2 = input("Enter your message 2: ")
input2 = "His 2 movies name."
# First message — sets initial context 
response2 = chat.send_message(input2) #This is from the Google Generative AI SDK 
print(response2.text)

for message in chat.get_history():
    print(f"{message.role.capitalize()  }: {message.parts[0].text}")    
#for message in chat.history:
 #   print(f"{message.role}: {message.content}")


