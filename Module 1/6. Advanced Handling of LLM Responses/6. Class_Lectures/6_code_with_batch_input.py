import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API Key not found. Set it in your .env file.")

# Initialize the Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", 
                              google_api_key=api_key)

# Create a list of prompts (batch input)
batch_prompts = [
    "Summarize the history of the internet in 3 lines.",
    "Explain what blockchain is for a beginner in 3 lines.",
    "What are the main components of a neural network in 3 lines?",
    "Give three use-cases of artificial intelligence in education  in 3 lines.",
    "Describe the importance of data privacy in healthcare systems  in 3 lines."
]

# Use .batch() to get results.  Sending whole list in one API call
responses = llm.batch(batch_prompts)

# Print responses
for i, response in enumerate(responses):
    print(f"\n🔹 Prompt {i+1}: {batch_prompts[i]}")
    print(f"🧠 Response {i+1}: {response.text}")
