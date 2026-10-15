"""
Book Library Assistant — Function / Tool Calling Demo
=======================================================
This is the core idea behind "function calling" (a.k.a. "tool calling"):

    1. We send the user's message to the model, along with a system prompt
       that describes the available tools and when/how to use them.
    2. The model replies in plain text. If it wants to use a tool, it
       replies in a special format we've told it to use:
           CALL_FUNCTION:function_name:param1=value,param2=value,...
       Otherwise it just replies normally (a conversational answer).
    3. We parse that text to detect a tool call (see utils/parser.py).
    4. If a tool call was found, we look up the matching Python function
       in a "tool registry" and run it with the parsed parameters.
    5. If no tool call was found, we show the model's text reply as-is.

Everything below is organized around that five-step flow.
"""

import os
from dotenv import load_dotenv
from google import genai
from utils.config import SYSTEM_PROMPT
from utils.parser import parse_function_call

load_dotenv()

# ---------------------------------------------------------------------------
# Allowed values for each enum-type parameter (used to validate tool input)
# ---------------------------------------------------------------------------
VALID_GENRES = ["fiction", "non_fiction", "mystery", "romance", "sci_fi", "biography", "history"]
VALID_ACTIONS = ["add", "remove", "view"]
VALID_STATUSES = ["reading", "completed", "paused", "abandoned"]

# ---------------------------------------------------------------------------
# A small mock book catalog so search_books / get_recommendations have real
# titles to return. In a real app this would be a database or an external
# books API instead.
# ---------------------------------------------------------------------------
BOOK_CATALOG = [
    {"id": "B001", "title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "genre": "fiction"},
    {"id": "B002", "title": "1984", "author": "George Orwell", "genre": "fiction"},
    {"id": "B003", "title": "To Kill a Mockingbird", "author": "Harper Lee", "genre": "fiction"},
    {"id": "B004", "title": "Murder on the Orient Express", "author": "Agatha Christie", "genre": "mystery"},
    {"id": "B005", "title": "The Hound of the Baskervilles", "author": "Arthur Conan Doyle", "genre": "mystery"},
    {"id": "B006", "title": "Gone Girl", "author": "Gillian Flynn", "genre": "mystery"},
    {"id": "B007", "title": "The Da Vinci Code", "author": "Dan Brown", "genre": "mystery"},
    {"id": "B008", "title": "And Then There Were None", "author": "Agatha Christie", "genre": "mystery"},
    {"id": "B009", "title": "Pride and Prejudice", "author": "Jane Austen", "genre": "romance"},
    {"id": "B010", "title": "The Notebook", "author": "Nicholas Sparks", "genre": "romance"},
    {"id": "B011", "title": "Me Before You", "author": "Jojo Moyes", "genre": "romance"},
    {"id": "B012", "title": "Dune", "author": "Frank Herbert", "genre": "sci_fi"},
    {"id": "B013", "title": "Ender's Game", "author": "Orson Scott Card", "genre": "sci_fi"},
    {"id": "B014", "title": "The Martian", "author": "Andy Weir", "genre": "sci_fi"},
    {"id": "B015", "title": "Foundation", "author": "Isaac Asimov", "genre": "sci_fi"},
    {"id": "B016", "title": "Sapiens", "author": "Yuval Noah Harari", "genre": "non_fiction"},
    {"id": "B017", "title": "Educated", "author": "Tara Westover", "genre": "biography"},
    {"id": "B018", "title": "Steve Jobs", "author": "Walter Isaacson", "genre": "biography"},
    {"id": "B019", "title": "Guns, Germs, and Steel", "author": "Jared Diamond", "genre": "history"},
    {"id": "B020", "title": "A People's History of the United States", "author": "Howard Zinn", "genre": "history"},
]


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def to_int(value):
    """Try to convert value to an int. Return None if that's not possible."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clamp_int(value, default, low, high):
    """Convert value to an int and clamp it into [low, high]; use default if invalid."""
    number = to_int(value)
    if number is None:
        number = default
    return max(low, min(number, high))


class LibraryAssistant:
    """
    Wraps a Gemini model with four "tools" it can call:
    search_books, manage_reading_list, get_recommendations, track_reading.
    """

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set GEMINI_API_KEY in .env")
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-3.6-flash"

        # In-memory reading list: this is where "add", "remove", "view", and
        # track_reading updates actually persist data for the life of this
        # LibraryAssistant instance. Key = normalized title/id, value = record.
        self.reading_list = {}

        # --- The tool registry ---
        # This dict is the heart of "function calling": it maps each tool
        # name the model is allowed to ask for, to the Python function that
        # actually carries it out. Adding a new tool later just means
        # writing a new method and adding one line here.
        self.tools = {
            "search_books": self._search_books,
            "manage_reading_list": self._manage_reading_list,
            "get_recommendations": self._get_recommendations,
            "track_reading": self._track_reading,
        }

    # -----------------------------------------------------------------
    # Step 1-2: talk to the model
    # -----------------------------------------------------------------
    def chat(self, user_input):
        """Send a message to the model and return the assistant's final reply."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[f"{SYSTEM_PROMPT}\n\nUser: {user_input}"],
            )
            reply_text = (response.text or "").strip()
        except Exception as e:
            return f"I'm sorry, I encountered an error while processing your request: {str(e)}"

        if not reply_text:
            return "❌ I didn't receive a valid response. Please try again."

        # Step 3: did the model ask to call a tool?
        function_name, parameters = parse_function_call(reply_text)

        if function_name is None:
            # No tool call -> the model answered conversationally, just show it.
            return reply_text

        # Step 4: run the requested tool.
        return self.handle_function_call(function_name, parameters)

    # -----------------------------------------------------------------
    # Step 4: dispatch to the right tool
    # -----------------------------------------------------------------
    def handle_function_call(self, function_name, parameters):
        """Look up `function_name` in the tool registry and run it."""
        tool = self.tools.get(function_name)
        if tool is None:
            return "❌ Sorry, I don't recognize that function."
        return tool(parameters)

    # -----------------------------------------------------------------
    # The tools themselves. Each one: validates its parameters, then
    # returns a text result the user will see.
    # -----------------------------------------------------------------

    def _search_books(self, parameters):
        query = parameters.get("query")
        genre = parameters.get("genre")
        max_results = parameters.get("max_results", 5)

        if not query or query == "None":
            return "❌ Search query is required."

        if genre and genre not in VALID_GENRES:
            return f"❌ Invalid genre: '{genre}'. Valid genres are: {', '.join(VALID_GENRES)}."

        max_results = clamp_int(max_results, default=5, low=1, high=20)

        query_lower = query.lower()
        matches = [
            book for book in BOOK_CATALOG
            if (not genre or book["genre"] == genre)
            and (
                query_lower in book["title"].lower()
                or query_lower in book["author"].lower()
                or query_lower == book["genre"]
            )
        ]

        # If the free-text query didn't match anything but a genre was
        # given, fall back to just showing books in that genre.
        if not matches and genre:
            matches = [b for b in BOOK_CATALOG if b["genre"] == genre]

        if not matches:
            return f"❌ No books found matching '{query}'."

        matches = matches[:max_results]
        lines = [f"{i + 1}. {b['title']} by {b['author']}" for i, b in enumerate(matches)]
        return "✅ Here are some books I found:\n" + "\n".join(lines)

    def _find_entry(self, identifier):
        """Find a reading-list entry by id or title (case-insensitive)."""
        if not identifier:
            return None
        identifier_lower = identifier.strip().lower()
        for key, record in self.reading_list.items():
            if key == identifier_lower:
                return key
            if record.get("id") and record["id"].lower() == identifier_lower:
                return key
            if record.get("title") and record["title"].lower() == identifier_lower:
                return key
        return None

    def _format_reading_list(self, status_filter=None):
        entries = list(self.reading_list.values())
        if status_filter:
            entries = [r for r in entries if r["status"] == status_filter]

        if not entries:
            if status_filter:
                return f"📭 No books with status '{status_filter.replace('_', ' ')}' yet."
            return "📭 Your reading list is empty. Try adding a book first!"

        lines = []
        for i, record in enumerate(entries, start=1):
            detail = (record["status"] or "not started").replace("_", " ")
            if record["rating"]:
                detail += f", rated {record['rating']}/5"
            elif record["pages_read"]:
                detail += f", {record['pages_read']} pages read"
            lines.append(f"{i}. {record['title']} — {detail}")

        heading = "✅ Here is your reading list:" if not status_filter else f"✅ Books marked '{status_filter}':"
        return heading + "\n" + "\n".join(lines)

    def _manage_reading_list(self, parameters):
        action = parameters.get("action")
        book_id = parameters.get("book_id")
        book_title = parameters.get("book_title")
        status_filter = parameters.get("status_filter")

        if action not in VALID_ACTIONS:
            return f"❌ Invalid action: '{action}'. Valid actions are: {', '.join(VALID_ACTIONS)}."

        if action == "view":
            if status_filter and status_filter not in VALID_STATUSES:
                return f"❌ Invalid status_filter: '{status_filter}'. Valid statuses are: {', '.join(VALID_STATUSES)}."
            return self._format_reading_list(status_filter)

        if not book_id and not book_title:
            return "❌ Please provide either a book_id or book_title for this action."

        identifier = book_id or book_title
        key = self._find_entry(identifier)

        if action == "add":
            if key is None:
                key = identifier.strip().lower()
                self.reading_list[key] = {
                    "id": book_id,
                    "title": book_title or book_id,
                    "status": None,
                    "pages_read": None,
                    "rating": None,
                }
            return f"✅ Added '{self.reading_list[key]['title']}' to your reading list."

        # action == "remove"
        if key is None:
            return f"❌ '{identifier}' isn't on your reading list."
        removed_title = self.reading_list.pop(key)["title"]
        return f"✅ Removed '{removed_title}' from your reading list."

    def _get_recommendations(self, parameters):
        genres = parameters.get("genres", [])
        authors = parameters.get("favorite_authors", [])
        count = parameters.get("count", 3)

        if not genres:
            return "❌ At least one genre is required."

        bad_genres = [g for g in genres if g not in VALID_GENRES]
        if bad_genres:
            return f"❌ Invalid genre(s): {', '.join(bad_genres)}. Valid genres are: {', '.join(VALID_GENRES)}."

        count = clamp_int(count, default=3, low=1, high=10)

        matches = [b for b in BOOK_CATALOG if b["genre"] in genres]

        if authors:
            authors_lower = [a.lower() for a in authors]
            matches.sort(key=lambda b: b["author"].lower() not in authors_lower)

        if not matches:
            return f"❌ No recommendations found for {', '.join(genres)}."

        matches = matches[:count]
        lines = [f"{i + 1}. {b['title']} by {b['author']}" for i, b in enumerate(matches)]
        return "✅ Here are some recommendations for you:\n" + "\n".join(lines)

    def _track_reading(self, parameters):
        book_id = parameters.get("book_id")
        status = parameters.get("status")
        pages_read = parameters.get("pages_read")
        rating = parameters.get("rating")

        if not book_id or book_id == "None":
            return "❌ book_id is required."

        if status not in VALID_STATUSES:
            return f"❌ Invalid status: '{status}'. Valid statuses are: {', '.join(VALID_STATUSES)}."

        if rating is not None:
            rating = to_int(rating)
            if rating is None or not (1 <= rating <= 5):
                return f"❌ Rating must be between 1 and 5. (Invalid rating: {rating})"

        if pages_read is not None:
            pages_read = to_int(pages_read)
            if pages_read is None or pages_read < 0:
                return "❌ Invalid pages_read: must be 0 or greater."

        # Find the matching reading-list entry by id or title; create one
        # if this book was never explicitly added first.
        key = self._find_entry(book_id)
        if key is None:
            key = book_id.strip().lower()
            self.reading_list[key] = {
                "id": book_id, "title": book_id, "status": None,
                "pages_read": None, "rating": None,
            }

        record = self.reading_list[key]
        record["status"] = status
        if pages_read is not None:
            record["pages_read"] = pages_read
        if rating is not None:
            record["rating"] = rating

        status_phrase = status.replace("_", " ")
        details = []
        if pages_read is not None:
            details.append(f"{pages_read} pages read")
        if rating is not None:
            details.append(f"rated {rating}/5")
        detail_phrase = f" ({', '.join(details)})" if details else ""

        return f"✅ Updated '{record['title']}' to {status_phrase}{detail_phrase}."
        detail_phrase = f" ({', '.join(details)})" if details else ""

        return f"✅ Updated '{book_id}' to {status_phrase}{detail_phrase}."


# ---------------------------------------------------------------------------
# Convenience wrapper + CLI demo
# ---------------------------------------------------------------------------

def answer_library_query(query, assistant=None):
    try:
        if assistant is None:
            assistant = LibraryAssistant()
        return assistant.chat(query)
    except Exception as e:
        return f"I'm sorry, I encountered an error while processing your request: {str(e)}"


def main():
    print("📚 Book Library Assistant - Function Calling Demo")
    print("Type 'quit' to exit")
    print("=" * 55)

    sample_queries = [
        "Find me some mystery books",
        "Add The Great Gatsby to my reading list",
        "Recommend me some fiction and sci-fi books",
        "I finished reading book B001, give it 4 stars",
        "Hello! How can you help me?",
        "What makes a good book?",
    ]

    assistant = LibraryAssistant()

    print("\n--- Sample Queries ---")
    for query in sample_queries:
        print(f"\nUser: {query}")
        print(f"Library: {answer_library_query(query, assistant)}")


if __name__ == "__main__":
    main()