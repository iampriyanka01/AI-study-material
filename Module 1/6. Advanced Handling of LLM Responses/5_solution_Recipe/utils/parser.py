import json


class ResponseParser:
    """Handles parsing of API responses and recipe data."""
    
    def parse_api_response(self, response_text):
        try:
            # Clean the response text
            cleaned_text = self.clean_response_text(response_text)
            
            # Parse JSON
            parsed = json.loads(cleaned_text)
            
            if "analysis" in parsed and isinstance(parsed["analysis"], list):
                return parsed["analysis"]
            
        except json.JSONDecodeError:
            pass
        
        # Return empty list if parsing fails
        return []
    
    def clean_response_text(self, text):
        cleaned = text.strip()
        
        # Remove markdown code fences if present
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
            
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
            
        return cleaned.strip()
    
