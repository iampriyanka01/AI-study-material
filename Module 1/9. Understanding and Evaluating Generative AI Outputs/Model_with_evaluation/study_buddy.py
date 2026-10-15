"""
study_buddy.py
===============
Everything related to the TARGET model ("Study Buddy").

This module owns:
- Creating a Study Buddy chat session (a fresh, stateful Gemini chat
  object configured with the Study Buddy system prompt)
- Sending a user query to that session and returning plain text

Both the Streamlit app (app.py) and the CLI (cli.py) call
`create_study_buddy_chat()` once per conversation and then
`get_study_buddy_response()` for every turn, so the "how do I talk to
the target model" logic exists in exactly one place.
"""

from google.genai import types

from config import client, STUDY_BUDDY_MODEL
from prompts import STUDY_BUDDY_SYSTEM_PROMPT


def create_study_buddy_chat():
    """
    Start a new, stateful Study Buddy chat session.

    Returns:
        A google.genai chat object. Keep a reference to it (e.g. in
        st.session_state) and reuse it across turns so Study Buddy
        remembers the conversation.
    """
    return client.chats.create(
        model=STUDY_BUDDY_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=STUDY_BUDDY_SYSTEM_PROMPT
        ),
    )


def get_study_buddy_response(study_chat, user_query: str) -> str:
    """
    Send one user message to an existing Study Buddy chat session.

    Args:
        study_chat: chat object returned by create_study_buddy_chat().
        user_query: the student's question.

    Returns:
        Study Buddy's reply as plain text.
    """
    response = study_chat.send_message(user_query)
    return response.text if hasattr(response, "text") else str(response)
