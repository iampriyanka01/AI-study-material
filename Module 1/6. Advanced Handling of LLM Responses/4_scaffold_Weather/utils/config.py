import os

def load_system_prompt() -> str:
    try:
        prompt_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompt.txt")
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError("prompt.txt file not found in project root.")
    except Exception as e:
        raise Exception(f"Error loading prompt.txt: {str(e)}")

def get_system_prompt() -> str:
    return load_system_prompt()

SYSTEM_PROMPT = get_system_prompt()
