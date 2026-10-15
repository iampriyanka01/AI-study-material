import os

def load_system_prompt():
    """Load the system prompt from prompt.txt file."""
    try:
        # Get the absolute path to the scaffold directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(base_dir, "..", "prompt.txt")

        # Open and read the prompt file with proper encoding
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        return content

    except FileNotFoundError:
        return "System prompt file not found. Please ensure prompt.txt exists in the scaffold directory."
    except Exception as e:
        return f"Error loading system prompt: {str(e)}"

def get_system_prompt():
    """Get the system prompt for the fitness coach."""
    return load_system_prompt()

# Export the system prompt
SYSTEM_PROMPT = get_system_prompt()
