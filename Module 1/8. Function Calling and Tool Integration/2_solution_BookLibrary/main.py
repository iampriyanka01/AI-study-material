import os
from dotenv import load_dotenv
from google import genai
from utils.config import SYSTEM_PROMPT, search_books_mock, manage_reading_list_mock, get_recommendations_mock, track_reading_mock
from utils.parser import parse_function_call, parse_array_parameter

load_dotenv()

class LibraryAssistant:
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.5-flash"
    
    def handle_function_call(self, function_name, parameters):
        """Handle library function call execution"""
        try:
            if function_name == "search_books":
                return self._handle_search_books(parameters)
            
            elif function_name == "manage_reading_list":
                return self._handle_reading_list(parameters)
            
            elif function_name == "get_recommendations":
                return self._handle_recommendations(parameters)
            
            elif function_name == "track_reading":
                return self._handle_track_reading(parameters)
            
            else:
                return f"Unknown function: {function_name}"
                
        except Exception as e:
            return f"Error executing function {function_name}: {str(e)}"
    
    def _handle_search_books(self, params):
        """Handle search_books function with parameter validation"""
        query = params.get("query", "")
        if not query:
            return "❌ Search query is required"
        
        genre = params.get("genre")
        max_results = params.get("max_results", 10)
        
        # Validate max_results range
        try:
            max_results = int(max_results)
            if max_results < 1 or max_results > 20:
                max_results = 10
        except (ValueError, TypeError):
            max_results = 10
        
        # Validate genre if provided
        valid_genres = ["fiction", "non_fiction", "mystery", "romance", "sci_fi", "biography", "history"]
        if genre and genre not in valid_genres:
            return f"❌ Invalid genre. Valid genres: {', '.join(valid_genres)}"
        
        result = search_books_mock(query, genre, max_results)
        
        if result["count"] == 0:
            return f"No books found matching '{query}'" + (f" in {genre} genre" if genre else "")
        
        response = f"📚 Found {result['count']} book(s):\n\n"
        for book in result["books"]:
            response += f"• **{book['title']}** by {book['author']}\n"
            response += f"  Genre: {book['genre'].replace('_', ' ').title()}, "
            response += f"Pages: {book['pages']}, Rating: {book['rating']}/5\n"
            response += f"  ID: {book['book_id']}\n\n"
        
        return response.strip()
    
    def _handle_reading_list(self, params):
        """Handle manage_reading_list function"""
        action = params.get("action")
        if not action:
            return "❌ Action is required (add, remove, or view)"
        
        valid_actions = ["add", "remove", "view"]
        if action not in valid_actions:
            return f"❌ Invalid action. Valid actions: {', '.join(valid_actions)}"
        
        if action == "view":
            result = manage_reading_list_mock(action)
            if result["count"] == 0:
                return "📖 Your reading list is empty. Add some books to get started!"
            
            response = f"📖 Your Reading List ({result['count']} books):\n\n"
            for book in result["books"]:
                response += f"• **{book['title']}** by {book['author']}\n"
                response += f"  Genre: {book['genre'].replace('_', ' ').title()}\n\n"
            
            return response.strip()
        
        # For add/remove actions
        book_id = params.get("book_id")
        book_title = params.get("book_title")
        
        if not book_id and not book_title:
            return f"❌ Book ID or title required for {action} action"
        
        result = manage_reading_list_mock(action, book_id, book_title)
        
        if result["success"]:
            return f"✅ {result['message']}"
        else:
            return f"❌ {result['message']}"
    
    def _handle_recommendations(self, params):
        """Handle get_recommendations function"""
        genres = params.get("genres", [])
        if not genres:
            return "❌ At least one genre is required for recommendations"
        
        # Parse genres array if it's a string
        if isinstance(genres, str):
            genres = parse_array_parameter(genres)
        
        # Validate genres
        valid_genres = ["fiction", "non_fiction", "mystery", "romance", "sci_fi", "biography", "history"]
        invalid_genres = [g for g in genres if g not in valid_genres]
        if invalid_genres:
            return f"❌ Invalid genres: {', '.join(invalid_genres)}. Valid genres: {', '.join(valid_genres)}"
        
        favorite_authors = params.get("favorite_authors", [])
        if isinstance(favorite_authors, str):
            favorite_authors = parse_array_parameter(favorite_authors)
        
        count = params.get("count", 5)
        try:
            count = int(count)
            if count < 1 or count > 10:
                count = 5
        except (ValueError, TypeError):
            count = 5
        
        result = get_recommendations_mock(genres, favorite_authors, count)
        
        if result["count"] == 0:
            return f"No recommendations found for {', '.join(genres)} genres"
        
        response = f"🌟 Book Recommendations ({result['count']} books):\n\n"
        for book in result["recommendations"]:
            response += f"• **{book['title']}** by {book['author']}\n"
            response += f"  Genre: {book['genre'].replace('_', ' ').title()}, "
            response += f"Rating: {book['rating']}/5\n"
            response += f"  ID: {book['book_id']}\n\n"
        
        return response.strip()
    
    def _handle_track_reading(self, params):
        """Handle track_reading function"""
        book_id = params.get("book_id")
        if not book_id:
            return "❌ Book ID is required for tracking"
        
        status = params.get("status")
        if not status:
            return "❌ Reading status is required"
        
        valid_statuses = ["reading", "completed", "paused", "abandoned"]
        if status not in valid_statuses:
            return f"❌ Invalid status. Valid statuses: {', '.join(valid_statuses)}"
        
        pages_read = params.get("pages_read")
        rating = params.get("rating")
        
        # Validate pages_read if provided
        if pages_read is not None:
            try:
                pages_read = int(pages_read)
                if pages_read < 0:
                    pages_read = 0
            except (ValueError, TypeError):
                pages_read = None
        
        # Validate rating if provided
        if rating is not None:
            try:
                rating = int(rating)
                if rating < 1 or rating > 5:
                    return "❌ Rating must be between 1 and 5 stars"
            except (ValueError, TypeError):
                return "❌ Rating must be a valid number"
        
        # Check if rating is only allowed for completed books
        if rating is not None and status != "completed":
            return "❌ Rating can only be provided for completed books"
        
        result = track_reading_mock(book_id, status, pages_read, rating)
        
        if result["success"]:
            return f"✅ {result['message']}"
        else:
            return f"❌ {result['message']}"
    
    def chat(self, user_input):
        """Main chat function that handles user input and decides whether to call functions"""
        try:
            # Prepare the conversation context
            conversation_context = f"{SYSTEM_PROMPT}\n\nUser: {user_input}\n\nAssistant:"
            
            # Call Gemini API
            response = self.client.models.generate_content(
                model=self.model,
                contents=conversation_context
            )
            
            if response.text is None or response.text.strip() == "":
                return "I didn't receive a valid response. Please try again."
            
            response_text = response.text.strip()
            
            # Check if response contains function call
            function_name, parameters = parse_function_call(response_text)
            
            if function_name and parameters is not None:
                # Execute the function call
                function_result = self.handle_function_call(function_name, parameters)
                return function_result
            else:
                # Return conversational response
                return response_text
                
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

def answer_library_query(query, assistant=None):
    """Convenient wrapper function for library queries"""
    try:
        if assistant is None:
            assistant = LibraryAssistant()
        
        response = assistant.chat(query)
        return response
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def interactive_mode():
    """Interactive library assistant mode"""
    print("Book Library Assistant - Interactive Mode")
    print("Type 'quit' to exit")
    print("=" * 55)
    
    assistant = LibraryAssistant()
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Thank you for using the library assistant. Happy reading!")
            break
        
        if not user_input:
            continue
        
        response = assistant.chat(user_input)
        print(f"\nLibrary: {response}")

def main():
    """Main function to demonstrate the library assistant."""
    print("Book Library Assistant - Intermediate Function Calling")
    print("=" * 55)
    
    # Sample test queries
    sample_queries = [
        "Find me some mystery books",
        "Add The Great Gatsby to my reading list",
        "Recommend me some fiction and sci-fi books",
        "I finished reading book B001, give it 4 stars",
        "Hello! How can you help me?",
        "What makes a good book?"
    ]
    
    assistant = LibraryAssistant()
    
    print("\n--- Sample Queries ---")
    for query in sample_queries:
        print(f"\nUser: {query}")
        response = answer_library_query(query, assistant)
        print(f"Library: {response}")
    
    print("\n" + "-" * 55)
    
    # Offer interactive mode
    print("\nWould you like to try interactive mode? (y/n)")
    if input().lower().startswith('y'):
        interactive_mode()

if __name__ == "__main__":
    main()