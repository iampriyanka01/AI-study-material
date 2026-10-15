from geminiClient import GeminiClient

class MenuAssistant:
    
    def __init__(self):
        self.client = GeminiClient()
    
    def answer(self, query: str) -> str:
        try:
            return self.client.call(query)
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"