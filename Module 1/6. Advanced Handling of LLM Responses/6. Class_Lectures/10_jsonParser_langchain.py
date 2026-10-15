import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API Key not found. Set it in your .env file.")

# Initialize the Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash",
                             google_api_key=api_key)

# Initialize parser
parser = JsonOutputParser()
format_instructions = parser.get_format_instructions()

resume_text = "My name is Prachi, I am twenty years old and my email is prachi@gmail.com"
prompt2=f"""
Extract the information in strict JSON format.
Text: {resume_text}

{format_instructions}
 """
response2 = llm.invoke(prompt2)
print(parser.parse(response2.text))