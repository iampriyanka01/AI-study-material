import json
import re

class ResponseParser:
    """Handles parsing of API responses and recipe data."""    
    def parse_api_response(self, response_text: str):
        try:
            # Clean the response text
            cleaned = self.clean_response_text(response_text)
            parsed = json.loads(cleaned)

            # Always return the full dict if it's a dic
            if isinstance(parsed, dict):
                return parsed
            
            # # If it's a list of recipes, wrap it
            if isinstance(parsed, list):
                return {"analysis": parsed, 
                        "batch_id": "batch_001", 
                        "total_recipes": len(parsed)}
            
            return {"batch_id": "batch_001", 
                    "total_recipes": 0, 
                    "analysis": []}
        
        except json.JSONDecodeError as e:
            return {"batch_id": "batch_001", 
                    "total_recipes": 0, 
                    "analysis": [], 
                    "error": f"JSON decode failed: {str(e)}"}
    
    def clean_response_text(self, text):
        cleaned = text.strip()
        
        # Remove leading code fences
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()

        # Remove trailing code fences
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        # Extract first JSON object or array if extra text exists
        match = re.search(r"(\{.*\}|  \[.*\])", cleaned, re.DOTALL)
        if match:
            return match.group(1).strip()
        return cleaned
