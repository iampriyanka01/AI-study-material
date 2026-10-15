"""
Configuration file for Weather Travel Advisor.
Handles loading the system prompt from prompt.txt file.
"""

import os


def load_system_prompt() -> str:
    """Load the system prompt from prompt.txt file."""
    try:
        prompt_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompt.txt')
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError("prompt.txt file not found. Please ensure it exists in the api_calling_2 directory")
    except Exception as e:
        raise Exception(f"Error loading prompt.txt: {str(e)}")


def get_system_prompt() -> str:
    """Get the system prompt with placeholders for weather data and user query."""
    return load_system_prompt()


# Export the system prompt
SYSTEM_PROMPT = get_system_prompt()
