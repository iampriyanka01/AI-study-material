"""
Simple Streamlit Chatbot with Auto-Summarization (Gemini API)
----------------------------------------------------------------
Every 5 user messages ("transactions"), the app generates a running
summary of the conversation so far and displays it above the chat.

Setup:
    pip install streamlit google-genai python-dotenv
    Create a .env file with: GEMINI_API_KEY=your-key-here
    streamlit run chatbot_app.py
"""

import os
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv, dotenv_values
from google import genai
from google.genai.errors import ServerError, ClientError, APIError

# Load .env from the same folder as this script, regardless of the
# directory Streamlit was launched from (a common source of "key not found").
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

MODEL = "gemini-3.6-flash"   # swap for another Gemini model ID if you prefer
SUMMARIZE_EVERY = 5  # number of user turns before summarizing

st.set_page_config(page_title="Chatbot with Auto-Summary", page_icon="💬")
st.title("💬 Chatbot with Auto-Summarization")

# --- Client setup ---------------------------------------------------------
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.sidebar.warning(
        f"No GEMINI_API_KEY found in `.env` at:\n\n`{env_path}`\n\n"
        "Checked for a file at that exact path. Make sure:\n"
        "- The file is named exactly `.env` (not `.env.txt`)\n"
        "- It sits next to `chatbot_app.py`\n"
        "- It contains a line like `GEMINI_API_KEY=your-key-here` "
        "(no quotes, no spaces around `=`)"
    )
    api_key = st.sidebar.text_input("Or paste your Gemini API Key", type="password")

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to start chatting.")
    st.stop()

try:
    if "client" not in st.session_state:
        st.session_state.client = genai.Client(api_key=api_key)
    client = st.session_state.client

    if "chat" not in st.session_state:
        st.session_state.chat = client.chats.create(model=MODEL)  # Gemini chat session
except ClientError as e:
    st.error(f"❌ Couldn't initialize the Gemini client — check your API key. Details: {e}")
    st.stop()
except ServerError:
    st.error("😕 Gemini's servers are unavailable right now. Please try again in a moment.")
    st.stop()

# --- Session state ----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []          # full chat history for display
if "user_turns" not in st.session_state:
    st.session_state.user_turns = 0          # count of user messages since last summary
if "summaries" not in st.session_state:
    st.session_state.summaries = []          # list of running summaries
if "last_summary_turn" not in st.session_state:
    st.session_state.last_summary_turn = None
if "last_summary_error" not in st.session_state:
    st.session_state.last_summary_error = None


def call_with_retry(fn, *, max_attempts=4, base_delay=2):
    """Call fn() with exponential backoff on transient server errors (e.g. 503
    UNAVAILABLE from high demand). Re-raises immediately on non-transient
    errors (e.g. bad API key, invalid request)."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except ServerError as e:
            if attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))  # 2s, 4s, 8s, ...
            st.toast(f"⚠️ Gemini is busy (attempt {attempt}/{max_attempts}). Retrying in {delay}s...")
            time.sleep(delay)
        except ClientError:
            # Not transient (e.g. bad key, bad request) — don't retry.
            raise
        except APIError:
            # Any other API error type — don't retry, just surface it.
            raise


def summarize_conversation(messages):
    """Summarize the conversation so far in a few sentences.

    Uses a one-off generate_content call (not the ongoing chat session) so the
    summarization request itself doesn't get added to the chat's history.
    """
    convo_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in messages
    )
    prompt = (
        "Summarize the following conversation in 3-4 concise sentences, "
        "capturing the key topics and any decisions or conclusions:\n\n"
        f"{convo_text}"
    )
    response = call_with_retry(
        lambda: client.models.generate_content(model=MODEL, contents=prompt)
    )
    return response.text


# --- Sidebar: show summaries -------------------------------------------------
with st.sidebar:
    st.header("📋 Running Summaries")
    if st.session_state.summaries:
        for i, s in enumerate(st.session_state.summaries, start=1):
            st.markdown(f"**Summary after turn {i * SUMMARIZE_EVERY}:**")
            st.write(s)
            st.divider()
    else:
        st.caption(f"A summary will appear here every {SUMMARIZE_EVERY} messages.")

    if st.button("🔄 Reset conversation"):
        st.session_state.messages = []
        st.session_state.chat = client.chats.create(model=MODEL)
        st.session_state.user_turns = 0
        st.session_state.summaries = []
        st.session_state.last_summary_turn = None
        st.session_state.last_summary_error = None
        st.rerun()

# --- Display chat history ----------------------------------------------------
if st.session_state.last_summary_turn is not None:
    st.success(f"📋 New summary generated after {st.session_state.last_summary_turn} messages — check the sidebar.")
    st.session_state.last_summary_turn = None  # show it only once

if st.session_state.last_summary_error is not None:
    st.warning(f"⚠️ The last summary attempt failed: {st.session_state.last_summary_error}")
    st.session_state.last_summary_error = None  # show it only once

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---------------------------------------------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # Show and store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.user_turns += 1

    # Get assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = call_with_retry(
                    lambda: st.session_state.chat.send_message(user_input)
                )
                reply = response.text
                st.markdown(reply)
            except ServerError:
                reply = None
                st.error(
                    "😕 Gemini is currently overloaded (503) and didn't respond "
                    "after several retries. Please try sending your message again "
                    "in a moment."
                )
            except ClientError as e:
                reply = None
                st.error(f"❌ Request failed: {e}")
            except Exception as e:
                reply = None
                st.error(f"⚠️ Unexpected error: {e}")

    if reply is not None:
        st.session_state.messages.append({"role": "assistant", "content": reply})

    # Every SUMMARIZE_EVERY user turns, generate a summary — this runs
    # regardless of whether the assistant reply above succeeded, so a single
    # failed/overloaded chat reply doesn't cause a summary to be skipped.
    if st.session_state.user_turns % SUMMARIZE_EVERY == 0:
        with st.spinner("Summarizing conversation..."):
            try:
                summary = summarize_conversation(st.session_state.messages)
                st.session_state.summaries.append(summary)
                st.session_state.last_summary_turn = st.session_state.user_turns
            except Exception as e:
                st.session_state.last_summary_error = str(e)
        st.rerun()
