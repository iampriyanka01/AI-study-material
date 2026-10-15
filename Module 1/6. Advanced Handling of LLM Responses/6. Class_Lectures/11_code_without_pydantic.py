from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import json

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API Key not found. Set it in your .env file.")


# Step 2: Create a raw prompt (instruct LLM to return JSON)
prompt = PromptTemplate(
    template= ("""
        Extract the following info in strict JSON format with keys: name, age, email.
        Text:{text}
        Respond only with JSON.
        """),
    input_variables=["text"]
)

# Step 1: Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=api_key
)

# Step 3: Create the chain manually (no parser)
chain = prompt | llm

# Step 4: Input to be processed
input_text = "Hi, my name is Priya Sharma. I'm very 20 and my email is priya.sharma@example.com."

# Step 5: Run and get the raw result
raw_response = chain.invoke({"text": input_text})
print("Raw LLM Output:\n", raw_response.content)
print("Raw LLM Output completed:\n")

# Step 6: Handle parsing safely
content = raw_response.content

if isinstance(content, str):
    # If it's a string, try parsing as JSON
    try:
        data = json.loads(content)
        print("\nParsed Data:\n", data)
    except json.JSONDecodeError as e:
        print("\n❌ JSON parsing failed:", e)

elif isinstance(content, list):
    # If it's a list, join or take first element
    try:
        combined = "".join(content) if all(isinstance(x, str) for x in content) else str(content[0])
        data = json.loads(combined)
        print("\nParsed Data:\n", data)
    except Exception as e:
        print("\n❌ Could not parse list content:", e)

else:
    print("\n⚠️ Unexpected type:", type(content))
