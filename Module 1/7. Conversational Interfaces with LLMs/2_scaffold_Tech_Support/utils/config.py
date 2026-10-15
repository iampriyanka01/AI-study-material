import os

def load_system_prompt():
    """Load the system prompt from prompt.txt file."""
    try:
        # Get the path to the directory where this config.py file lives
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Build the full path to the prompt.txt file in the scaffold directory
        prompt_path = os.path.join(base_dir, "prompt.txt")

        # Open and read the prompt file with proper UTF-8 encoding
        with open(prompt_path, "r", encoding="utf-8") as f:
            # Return the content stripped of leading/trailing whitespace
            return f.read().strip()

    except FileNotFoundError:
        # Handle FileNotFoundError with appropriate message
        return "System prompt file not found. Please create prompt.txt."

    except Exception as e:
        # Handle other exceptions with appropriate message
        return f"Error loading system prompt: {e}"


def get_system_prompt():
    """Get the system prompt for the technical support assistant."""
    # Call load_system_prompt() and return the result
    return load_system_prompt()


# Export the system prompt as SYSTEM_PROMPT
SYSTEM_PROMPT = get_system_prompt()
