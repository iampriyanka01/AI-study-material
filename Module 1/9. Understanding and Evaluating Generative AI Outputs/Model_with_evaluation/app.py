"""
app.py
======
Streamlit front end for the Study Buddy + Evaluator pipeline.

Flow, every turn:
    1. Student types a question in the chat box.
    2. study_buddy.get_study_buddy_response() sends it to the TARGET
       model and returns the answer.
    3. evaluator.evaluate_response() sends (question, answer) to the
       EVALUATOR model and returns a scores/feedback/summary verdict.
    4. Both the answer and the verdict are saved into st.session_state
       and rendered: Study Buddy's answer as a normal chat bubble,
       immediately followed by an "Evaluator feedback" panel.

This file contains ONLY Streamlit/UI code. All Gemini-related logic
(prompts, API calls, JSON parsing) lives in prompts.py, study_buddy.py
and evaluator.py, so this file stays easy to read and the underlying
pipeline can be reused (e.g. by cli.py) without touching Streamlit.

Run with:
    streamlit run app.py
"""

import streamlit as st

from study_buddy import create_study_buddy_chat, get_study_buddy_response
from evaluator import create_evaluator_chat, evaluate_response

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="Study Buddy", page_icon="📚", layout="centered")
st.title("📚 Study Buddy")
st.caption(
    "Ask a school question. Every answer is automatically graded by a "
    "second 'evaluator' model on accuracy, clarity, tone, relevance and "
    "conciseness."
)


# ------------------------------------------------------------------
# Session state: keeps chat objects + message history alive across
# Streamlit reruns (Streamlit reruns the whole script on every
# interaction, so anything that must persist has to live here).
# ------------------------------------------------------------------
def init_session_state():
    """Create the chat sessions and empty history exactly once."""
    if "study_chat" not in st.session_state:
        st.session_state.study_chat = create_study_buddy_chat()
    if "evaluator_chat" not in st.session_state:
        st.session_state.evaluator_chat = create_evaluator_chat()
    if "messages" not in st.session_state:
        # Each item: {"role": "user"/"assistant", "content": str,
        #             "evaluation": dict | None, "eval_error": str | None}
        st.session_state.messages = []


init_session_state()


# ------------------------------------------------------------------
# Rendering helpers
# ------------------------------------------------------------------
def render_evaluation_panel(evaluation: dict | None, eval_error: str | None):
    """
    Render the evaluator's verdict directly under a Study Buddy answer.

    Shows:
        - a metric per rubric dimension (score out of 5)
        - a short feedback comment per dimension
        - the overall 2-3 sentence summary
    Falls back to a warning if evaluation failed or is missing.
    """
    with st.expander("🧪 Evaluator feedback", expanded=True):
        if eval_error:
            st.warning(f"Evaluation unavailable: {eval_error}")
            return
        if not evaluation:
            st.info("No evaluation available for this response.")
            return

        scores = evaluation.get("scores", {})
        feedback = evaluation.get("feedback", {})
        summary = evaluation.get("overall_summary", "")

        # Scores as a row of metrics (out of 5.0)
        dimensions = ["accuracy", "clarity", "coherence", "relevance", "tone", "conciseness"]
        present_dims = [d for d in dimensions if d in scores]
        if present_dims:
            cols = st.columns(len(present_dims))
            for col, dim in zip(cols, present_dims):
                col.metric(label=dim.capitalize(), value=f"{scores[dim]:.1f} / 5")

        # Per-dimension feedback comments
        if feedback:
            st.markdown("**Feedback details:**")
            for dim in present_dims:
                if dim in feedback:
                    st.markdown(f"- **{dim.capitalize()}:** {feedback[dim]}")

        # Overall summary
        if summary:
            st.markdown("**Overall summary:**")
            st.markdown(summary)


def render_message_history():
    """Replay every stored message (and, for assistant turns, its evaluation)."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_evaluation_panel(msg.get("evaluation"), msg.get("eval_error"))


# ------------------------------------------------------------------
# Main render pass: show history first, then handle new input
# ------------------------------------------------------------------
render_message_history()

user_query = st.chat_input("Ask a school question...")

if user_query:
    # 1) Show the student's message immediately
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2) Get Study Buddy's (target model's) answer
    with st.chat_message("assistant"):
        with st.spinner("Study Buddy is thinking..."):
            try:
                target_output = get_study_buddy_response(
                    st.session_state.study_chat, user_query
                )
            except Exception as e:
                target_output = f"⚠️ Study Buddy failed to respond: {e}"
        st.markdown(target_output)

        # 3) Get the evaluator's verdict on that answer
        evaluation, eval_error = None, None
        with st.spinner("Evaluator is grading the response..."):
            try:
                evaluation = evaluate_response(
                    st.session_state.evaluator_chat, user_query, target_output
                )
            except Exception as e:
                eval_error = str(e)

        render_evaluation_panel(evaluation, eval_error)

    # 4) Persist this full turn (answer + evaluation) so it survives reruns
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": target_output,
            "evaluation": evaluation,
            "eval_error": eval_error,
        }
    )
