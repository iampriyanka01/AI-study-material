import os
import streamlit as st
from main import LibraryAssistant

st.set_page_config(page_title="Book Library Assistant", page_icon="📚", layout="centered")

st.title("📚 Book Library Assistant")
st.caption("Search books, manage your reading list, get recommendations, and track your progress.")

SAMPLE_QUERIES = [
    "Find me some mystery books",
    "Add The Great Gatsby to my reading list",
    "Recommend me some fiction and sci-fi books",
    "I finished reading book B001, give it 4 stars",
    "Hello! How can you help me?",
    "What makes a good book?",
]

STATUS_EMOJI = {
    "reading": "📖",
    "completed": "✅",
    "paused": "⏸️",
    "abandoned": "🚫",
    None: "⬜",
}

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state["messages"] = []


def get_assistant():
    """Lazily create (and cache) the LibraryAssistant for this session."""
    if "assistant" in st.session_state:
        return st.session_state["assistant"]
    try:
        assistant = LibraryAssistant(api_key=os.getenv("GEMINI_API_KEY") or None)
        st.session_state["assistant"] = assistant
        st.session_state.pop("assistant_error", None)
        return assistant
    except ValueError as e:
        st.session_state["assistant_error"] = str(e)
        return None


assistant = get_assistant()

# ---------- Sidebar: API key + reading list + sample queries ----------
with st.sidebar:
    st.header("Settings")
    key_input = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Loaded from .env by default. Paste one here to override it for this session.",
    )
    if st.button("Save API Key", use_container_width=True):
        os.environ["GEMINI_API_KEY"] = key_input
        st.session_state.pop("assistant", None)
        st.session_state.pop("assistant_error", None)
        st.success("Saved for this session.")
        st.rerun()

    st.divider()
    st.header("📖 Your Reading List")
    if assistant is None:
        st.caption("Set your API key above to start tracking books.")
    elif not assistant.reading_list:
        st.caption("Nothing here yet — try a sample query like 'Add The Great Gatsby to my reading list'.")
    else:
        for record in assistant.reading_list.values():
            emoji = STATUS_EMOJI.get(record["status"], "⬜")
            status_label = (record["status"] or "not started").replace("_", " ").title()
            detail = status_label
            if record["rating"]:
                detail += f" · ⭐ {record['rating']}/5"
            elif record["pages_read"]:
                detail += f" · {record['pages_read']} pages"
            st.markdown(f"{emoji} **{record['title']}**  \n<span style='color:gray'>{detail}</span>", unsafe_allow_html=True)

        if st.button("Clear reading list", use_container_width=True):
            assistant.reading_list.clear()
            st.rerun()

    st.divider()
    st.header("Try a sample query")
    for q in SAMPLE_QUERIES:
        if st.button(q, use_container_width=True, key=f"sample_{q}"):
            st.session_state["pending_query"] = q

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ---------- Render chat history ----------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Get the next user message (sidebar sample click OR typed input) ----------
user_query = st.session_state.pop("pending_query", None)
typed_query = st.chat_input("Ask about books, your reading list, recommendations, or progress...")
if typed_query:
    user_query = typed_query

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if assistant is None:
            error_msg = st.session_state.get("assistant_error", "Gemini API key not set.")
            reply = f"⚠️ {error_msg}\n\nAdd your GEMINI_API_KEY in the sidebar to get started."
            st.markdown(reply)
        else:
            with st.spinner("Thinking..."):
                reply = assistant.chat(user_query)
            st.markdown(reply)

    st.session_state["messages"].append({"role": "assistant", "content": reply})
    # Rerun so the sidebar reading-list panel reflects any add/remove/track
    # that just happened during this turn.
    st.rerun()

if not st.session_state["messages"]:
    st.info("👋 Start by typing a question below, or click a sample query in the sidebar.")