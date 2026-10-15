"""
cli.py
======
Optional terminal version of the same pipeline, for anyone who prefers
a plain interactive prompt over the Streamlit UI.

Note this file contains almost no logic of its own — it just wires
together study_buddy.py and evaluator.py, exactly like app.py does.
That's the point of the refactor: the "talk to Study Buddy" and "grade
the answer" logic each live in exactly one place, and both the web UI
and this CLI call the same functions.

Run with:
    python cli.py
"""

import json

from study_buddy import create_study_buddy_chat, get_study_buddy_response
from evaluator import create_evaluator_chat, evaluate_response


def main():
    print("Study Buddy + Evaluator (CLI mode). Type 'exit' to quit.\n")

    # One chat session per model, reused for the whole conversation.
    study_chat = create_study_buddy_chat()
    evaluator_chat = create_evaluator_chat()

    while True:
        user_query = input("You: ").strip()
        if not user_query:
            print("Please type a question, or 'exit' to quit.")
            continue
        if user_query.lower() in ("exit", "quit"):
            print("Study Buddy: Goodbye! Have a great time learning. 😊")
            break

        # --- Target model turn ---
        try:
            target_output = get_study_buddy_response(study_chat, user_query)
        except Exception as e:
            print(f"[Study Buddy error] {e}")
            continue

        print(f"\nStudy Buddy: {target_output}\n")

        # --- Evaluator turn ---
        try:
            evaluation = evaluate_response(evaluator_chat, user_query, target_output)
            print("--- Evaluator feedback ---")
            print(json.dumps(evaluation, indent=2))
            print("--------------------------\n")
        except Exception as e:
            print(f"[Evaluator error] {e}\n")


if __name__ == "__main__":
    main()
