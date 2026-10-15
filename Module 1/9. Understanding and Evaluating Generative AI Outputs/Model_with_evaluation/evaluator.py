"""
evaluator.py
=============
Everything related to the EVALUATOR model ("LLM-as-judge").

This module owns:
- Creating an evaluator chat session (a separate Gemini instance,
  independent from Study Buddy, configured with the rubric prompt)
- Sending it a (user_query, target_output) pair and getting back a
  score/feedback verdict
- Safely parsing that verdict out of the model's raw text, even if the
  model wraps the JSON in markdown fences or minor formatting slips

Both app.py and cli.py call `create_evaluator_chat()` once and then
`evaluate_response()` for every turn, so the grading logic lives here
and only here.
"""

import json
import re

from google.genai import types

from config import client, EVALUATOR_MODEL
from prompts import EVALUATOR_SYSTEM_PROMPT

# Keys every evaluator verdict must contain. Used for validation so a
# malformed model response fails loudly instead of silently.
REQUIRED_KEYS = ("scores", "feedback", "overall_summary")


def create_evaluator_chat():
    """
    Start a new, stateful evaluator chat session.

    Kept separate from the Study Buddy session on purpose: the evaluator
    should judge each answer independently, without its own chat history
    influencing the score.
    """
    return client.chats.create(
        model=EVALUATOR_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=EVALUATOR_SYSTEM_PROMPT
        ),
    )


def safe_parse_json(text: str) -> dict:
    """
    Extract the first balanced {...} block from `text` and parse it as
    JSON, tolerating common small mistakes (single quotes, trailing
    commas) that a model might introduce.

    Raises:
        ValueError if no valid JSON object could be found/parsed.
    """
    stack = []
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        # Try to repair common minor issues before giving up
                        # on this candidate and looking for the next one.
                        fixed = candidate.replace("'", '"')
                        fixed = re.sub(r",\s*}", "}", fixed)
                        fixed = re.sub(r",\s*]", "]", fixed)
                        try:
                            return json.loads(fixed)
                        except Exception:
                            pass
                    start = None
    raise ValueError("No valid JSON object found in evaluator output.")


def evaluate_response(evaluator_chat, user_query: str, target_output: str) -> dict:
    """
    Ask the evaluator model to grade one Study Buddy answer.

    Args:
        evaluator_chat: chat object returned by create_evaluator_chat().
        user_query: the original student question.
        target_output: Study Buddy's answer to that question.

    Returns:
        A dict with keys "scores", "feedback", "overall_summary".

    Raises:
        ValueError if the evaluator's reply couldn't be parsed as the
        expected JSON shape.
    """
    eval_input = (
        f"Original Prompt (User Query):\n{user_query}\n\n"
        f"Target Model's Output:\n{target_output}\n\n"
        f"Please evaluate using the rubric and return ONLY the JSON specified."
    )

    eval_resp = evaluator_chat.send_message(eval_input)
    eval_text = eval_resp.text if hasattr(eval_resp, "text") else str(eval_resp)

    try:
        eval_json = safe_parse_json(eval_text)
    except ValueError as e:
        raise ValueError(
            f"Failed to parse evaluator JSON: {e}\n\nRaw evaluator output:\n{eval_text}"
        )

    missing = [k for k in REQUIRED_KEYS if k not in eval_json]
    if missing:
        raise ValueError(f"Evaluator JSON missing required key(s): {missing}")

    return eval_json
