from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API Key not found. Set it in your .env file.")

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", 
                             google_api_key=api_key)

# Simple query
#core function in LangChain used to run a chain, model, or tool once with a single input and return the corresponding output.
response = llm.invoke("Explain quantum computing in simple terms of 50 words.")
print(f'Response: {response.text}')