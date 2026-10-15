from llmClient import LLMClient

class MenuAssistant:
    """Assistant that answers restaurant-related queries using an LLM."""    
    def __init__(self):
        self.client = LLMClient()

    def answer(self, query: str) -> str:
        """Answer user question using the LLM and the prompt."""
        try:
            response = self.client.call_api(query)
            return response
        except Exception as e:
            return "Sorry, I couldn’t process that request. Please try again."