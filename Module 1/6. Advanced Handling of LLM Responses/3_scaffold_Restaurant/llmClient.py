import os
from google import genai
from utils.config import SYSTEM_PROMPT
from typing import Optional
import json


class LLMClient:
    """Client for interacting with Gemini API for campus event Q&A."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in environment variables.")

        self.client = genai.Client(api_key=self.api_key)
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    
    def call_api(self, user_query: str) -> str:
        """Call Gemini API with system prompt and user query."""

        full_prompt = SYSTEM_PROMPT.format(user_query=user_query)
                
        # TODO: Make API call to llm
        try:
            response = self.client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=full_prompt
                    )
        except Exception as e:
            return f"API call failed: {str(e)}"
        
        # TODO: Handle empty or None response
        if response.text is None or response.text.strip() == "":
            return "I didn't receive a valid response. Please try again."
        else:            
            try:                
                return json.loads(response.text.replace("```json", "").replace("```python", "").replace("```", "").strip())      
            except:
                return response.text

