import os

def load_system_prompt():
    """Load the system prompt from prompt.txt file."""
    try:
        # Look for prompt.txt in the solution directory (same level as main.py)
        prompt_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompt.txt')
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError("prompt.txt file not found. Please ensure it exists in the solution directory")
    except Exception as e:
        raise Exception(f"Error loading prompt.txt: {str(e)}")

def get_system_prompt():
    """Get the system prompt for the fitness coach."""
    return load_system_prompt()

# Export the system prompt
SYSTEM_PROMPT = get_system_prompt()
