import os
from google import genai
from google.genai.types import GenerateContentConfig
from dotenv import load_dotenv
import json

# Step 1: Initialize the Gemini client
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Step 2: Define system instruction & generation config
config = GenerateContentConfig(
    system_instruction=[
        "As a tokenizer when you are given an input sentence, output a Python-style list of tokens, "
        "splitting words, subwords (e.g., contractions), and punctuation."
    ],
    max_output_tokens=1000,
    temperature=0.0, # deterministic output
    top_p=1.0,
    top_k=1
)

# Step 3: Hardcoded sentence
#input_sentence = "I can't believe it's not butter!"
input_sentence = input("Enter a sentence: ")

# Step 4: Send a single-turn prompt to generate_content
response = client.models.generate_content(
    model='gemini-3.6-flash',
    contents=f"Tokenize into a Python list, give me all the words:\n\n{input_sentence}",
    config=config
)

token_str = response.candidates[0].content.parts[0].text.strip()

# Remove Markdown fences if present
token_str = token_str.replace("```json", "").replace("```python", "").replace("```", "").strip()

if not token_str:
    print("❌ No output returned from Gemini")
    tokens = []
else:
    try:
        tokens = json.loads(token_str)
        print("Original Sentence:", input_sentence)
        print("Tokens:", tokens)
        print("Total Tokens:", len(tokens))
    except json.JSONDecodeError as e:
        print("❌ JSON parsing failed:", e)
        print("Raw output was:", token_str)
        tokens = []