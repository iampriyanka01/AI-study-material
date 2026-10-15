import os

def load_system_prompt() -> str:
    """Load the system prompt from prompt.txt file."""
    try:
    # TODO: Get the path to prompt.txt file
        Prompt_dir = os.path.dirname(os.path.dirname(__file__))
        prompt_file = os.path.join(Prompt_dir, 'prompt.txt')
        
        # TODO: Open and read the prompt file
        with open(prompt_file, 'r', encoding='utf-8') as f:
                    
        # TODO: Return the content stripped of whitespace
            return f.read().strip()
        # TODO: Handle FileNotFoundError with appropriate message
    except FileNotFoundError:
            raise FileNotFoundError("prompt.txt file not found. Please ensure it exists in the api_calling directory")
    # TODO: Handle other exceptions with appropriate message
    except Exception as e:
        raise Exception(f"Error loading prompt.txt: {str(e)}")


def get_system_prompt() -> str:
    """Get the system prompt with placeholders for weather data and user query."""
    # TODO: Call load_system_prompt() and return the result
    return load_system_prompt()


# TODO: Export the system prompt as SYSTEM_PROMPT
SYSTEM_PROMPT = get_system_prompt()