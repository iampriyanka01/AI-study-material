import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

def main():

    load_dotenv()   
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API Key not found. Please set it in your .env file.")

    print("--- 🚀 Initializing Model and Chain ---")    
    # 2. Instantiate the Gemini Model for Streaming

    # Initialize model directly
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        streaming=True # is good practice to signal our intent.
    )

    # Input message
    messages = [
        {"role": "user", 
         "content": "Explain black holes in simple terms in about 200 words."}
    ]

    print("🤖 Generating response...\n")

    try:
        # Instead of .invoke(), we use .stream() to get a generator for streaming output
        # response = llm.invoke(messages)
        response = llm.stream(messages)
        for chunk in response:
            for part in chunk.content:   # iterate over each dict in the list
                if part["type"] == "text":
                    print(part["text"], end="", flush=True)
                     # By setting end="", you tell Python not to add anything extra (default new line) after the text.
                     #Setting flush=True forces Python to immediately write the output to the console instead of waiting. This is important for real-time streaming responses
    except Exception as e:
        print(f"\n\nAn error occurred: {e}")


    print("\n\n✅ Stream Complete.")
    print("This script demonstrates stream() method is synchronous method because this will be executed after the streaming completes. ")

if __name__ == "__main__":
    main()
