import os

def load_system_prompt() -> str:
    """Load the system prompt from prompt.txt file."""
    try:
        # TODO: Get the path to prompt.txt file in the parent directory
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompt.txt")
        # TODO: Open and read the prompt file with proper encoding
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        # TODO: Handle FileNotFoundError with appropriate message
        raise FileNotFoundError("prompt.txt file not found in project root.")
    except Exception as e:
        # TODO: Handle other exceptions with appropriate message
        raise RuntimeError(f"Error loading system prompt: {str(e)}")

def get_system_prompt() -> str:
    """Get the system prompt with recipes placeholder."""
    # TODO: Call load_system_prompt() and return the result
    return load_system_prompt()

# TODO: Export the system prompt as SYSTEM_PROMPT
SYSTEM_PROMPT = get_system_prompt()
