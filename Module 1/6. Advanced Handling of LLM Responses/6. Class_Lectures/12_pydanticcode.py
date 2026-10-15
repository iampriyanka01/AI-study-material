from langchain_google_genai import ChatGoogleGenerativeAI # lets you talk to Gemini.
from langchain_core.output_parsers import PydanticOutputParser # makes sure Gemini’s answer matches a Pydantic model.
from pydantic import BaseModel, EmailStr # blueprint for your data. special type that validates emails.
from dotenv import load_dotenv
import os

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API Key not found. Set it in your .env file.")

# Step 1: Define the Pydantic Model
#This says: “I expect a person with a name (string), age (integer), and email (valid email).
class Person(BaseModel):
    name: str
    age: int
    email: EmailStr

# Step 2: Create the Output Parser
#This parser will take Gemini’s messy text and try to turn it into a Person object.
parser = PydanticOutputParser(pydantic_object=Person)
format_instructions = parser.get_format_instructions() #tells Gemini: “Output must look like JSON that matches the Person model.”


# Step 3: Manually format prompt string (no PromptTemplate)
resume_text = "Hi, my name is Priya Sharma. I'm twenty and my email is priya.sharma@example.com."

final_prompt = f"""Extract person info from the following text:
Text: {resume_text}
{format_instructions}
"""

# Step 4: Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=api_key
)

# Step 5: Get LLM response
response = llm.invoke(final_prompt)

# Step 6: Normalize content into a string
content = response.content
if isinstance(content, list):    
    content = content[0].get("text", "") ## Extract the text field from the first chunk

# Step 7: Parse and validate
try:
    print(f"✅ Structured Output: {parser.parse(content)}")
except Exception as e:
    print(f"❌ Validation failed: {e}")

