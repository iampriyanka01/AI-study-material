from typing import Optional
from google import genai
from utils.config import SYSTEM_PROMPT
import json
import os

class GeminiClient:
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        self.client = genai.Client(api_key=self.api_key)
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    def call(self, user_query: str) -> str:
        full_prompt = SYSTEM_PROMPT.format(user_query=user_query)
        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt
        )
        
        if response.text is None or response.text.strip() == "":
            return "I didn't receive a valid response from the model. Please try again."
        
        # Strip markdown fences if present
        content_stripped = response.text.strip()
        if content_stripped.startswith("```"):
            lines = content_stripped.splitlines()
            # Remove first and last fence lines if present
            if lines[0].startswith("```") and lines[-1].startswith("```"):
                content_stripped = "\n".join(lines[1:-1]).strip()
        return content_stripped