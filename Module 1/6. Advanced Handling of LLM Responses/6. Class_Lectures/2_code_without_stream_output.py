import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API Key not found. Please set it in your .env file.")

    print("--- 🚀 Initializing Model (No Streaming) ---")
    # Initialize model WITHOUT streaming
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash", 
        streaming=False  #  Turn off streaming
    )   

    # Input message
    messages = [
         {"role": "user", 
         "content": "Explain black holes in simple terms in about 2000 words."}] 

    try:
        response = llm.invoke(messages)
        print(response.text)
    except Exception as e:
        print(f"\n\nAn error occurred: {e}")
    print("\n\n✅ Response Complete.")

if __name__ == "__main__":
    main()
