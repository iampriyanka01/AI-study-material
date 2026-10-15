import google.genai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# ——— Configuration ———
API_KEY = os.getenv("GOOGLE_API_KEY")

# Create the Gemini client
client = genai.Client(api_key=API_KEY)
MODEL = "gemini-3.6-flash"
SUMMARY_THRESHOLD = 6

# ——— Helper: Summarize chat history ———
def summarize_history(history: list[dict]) -> str:
    transcript = "\n".join(f"{m['role']}: {m['text']}" for m in history)
    prompt = f"""
You are a conversation summarizer. Summarize the chat below, capturing main points and intent.

{transcript}
--
Provide a concise summary:"""

    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    return resp.text.strip()

# ——— Main chat loop ———
def run_chat():
    history = []
    print("🧠 Gemini Chat — type 'quit' to exit.")

    while True:
        userinput = input("You: ")
        if userinput.lower() in ("quit", "exit"):
            print("Session ended.")
            break

        history.append({"role": "user", "text": userinput})

        # Summarize if threshold exceeded
        if len(history) >= SUMMARY_THRESHOLD:
            summary = summarize_history(history)
            history = [
                {"role": "system", "text": f"Summary of previous convo: {summary}"},
            ]
            print("\n--- [SYSTEM] Conversation summarized. ———\n")
            print(f"Summary: {summary}\n")

        # Send chat prompt
        transcript = "\n".join(f"{m['role']}: {m['text']}" for m in history)
        print(f"\n--- [DEBUG] Current transcript:\n{transcript}\n")
        mresp = client.models.generate_content(
            model=MODEL,
            contents=transcript + "\nassistant:"
        )
        assistant_msg = mresp.text.strip()

        print("Gemini:", assistant_msg)
        history.append({"role": "assistant", "text": assistant_msg})

if __name__ == "__main__":
    run_chat()
