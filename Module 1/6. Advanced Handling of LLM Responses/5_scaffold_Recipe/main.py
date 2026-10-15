import json
import os
from dotenv import load_dotenv
from google import genai
# TODO: Import SYSTEM_PROMPT from utils.config
from utils.config import SYSTEM_PROMPT
# TODO: Import ResponseParser from utils.parser
from utils.parser import ResponseParser


# TODO: Load environment variables
load_dotenv()

class RecipeBatchAnalyzer:
    """Analyzes multiple recipes in a single batch using Gemini API."""
    
    def __init__(self, api_key=None):
        # TODO: Initialize API key from parameter or env var GEMINI_API_KEY
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        # TODO: Raise ValueError if no key found
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        # TODO: Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)

        # TODO: Set model from env var or default to "gemini-3.6-flash"
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

        # TODO: Initialize ResponseParser
        self.parser = ResponseParser()
    
    def format_recipes_for_analysis(self, recipes):
        """Format recipes for batch analysis."""
        # TODO: Create a list to store formatted recipes
        formatted = []
        # TODO: Loop through recipes with enumerate starting from 1
        for i, recipe in enumerate(recipes, start=1):
            # TODO: Format each recipe with number, name, ingredients, instructions
            name = recipe.get("name", f"Recipe {i}")
            ingredients = ", ".join(recipe.get("ingredients", [])) or "No ingredients provided"
            instructions = recipe.get("instructions", "No instructions provided")
            formatted.append(
                f"Recipe {i}: {name}\nIngredients: {ingredients}\nInstructions: {instructions}\n"
            )
        # TODO: Return joined formatted recipes
        return "\n".join(formatted)
        '''
        Recipe 1: Spaghetti Carbonara
        Ingredients: spaghetti, eggs, bacon, parmesan, black pepper
        Instructions: Cook pasta, fry bacon, mix with eggs and cheese

        Recipe 2: Caesar Salad
        Ingredients: lettuce, croutons, parmesan, caesar dressing
        Instructions: Chop lettuce, add toppings, toss with dressing

        Recipe 3: Chocolate Chip Cookies
        Ingredients: flour, butter, sugar, chocolate chips, eggs
        Instructions: Mix ingredients, form cookies, bake at 350F
        '''
    
    def analyze_batch(self, recipes):
        if not recipes:
            return {"batch_id": "batch_000", 
                    "total_recipes": 0, 
                    "analysis": []}
        # Limit to 5 recipes for API efficiency
        recipes = recipes[:5]

        # Format recipes
        formatted_recipes = self.format_recipes_for_analysis(recipes)

        # Create prompt
        prompt = SYSTEM_PROMPT.format(recipes=formatted_recipes)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            # Parse response using parser
            parsed = self.parser.parse_api_response(response.text)

            # Return parsed JSON directly
            return parsed
        except Exception as e:
            return {
                "batch_id": "batch_001",
                "total_recipes": len(recipes),
                "analysis": [],
                "error": str(e)
            }



def main():
    """Main function to demonstrate the recipe analyzer."""
    sample_recipes = [
        {
            "name": "Spaghetti Carbonara",
            "ingredients": ["spaghetti", "eggs", "bacon", "parmesan", "black pepper"],
            "instructions": "Cook pasta, fry bacon, mix with eggs and cheese"
        },
        {
            "name": "Caesar Salad",
            "ingredients": ["lettuce", "croutons", "parmesan", "caesar dressing"],
            "instructions": "Chop lettuce, add toppings, toss with dressing"
        },
        {
            "name": "Chocolate Chip Cookies",
            "ingredients": ["flour", "butter", "sugar", "chocolate chips", "eggs"],
            "instructions": "Mix ingredients, form cookies, bake at 350F"
        }
    ]
    
    print("Recipe Batch Analyzer")
    print("=" * 50)
    
    analyzer = RecipeBatchAnalyzer()
    
    try:
        result = analyzer.analyze_batch(sample_recipes)
        print(f"Analyzed {result.get('total_recipes', 0)} recipes\n")
        
        # ✅ Print full JSON
        print("Results (JSON):")
        print(json.dumps(result, indent=2))

        '''
        # ✅ Print summary
        print("\nResults (Summary):")
        for recipe in result.get('analysis', []):
            name = recipe.get('recipe_name', 'Unknown')
            difficulty = recipe.get('difficulty', 'unknown')
            time = recipe.get('cooking_time', 'unknown')
            calories = recipe.get('calories_per_serving', 'unknown')
            print(f"- {name}: {difficulty} difficulty, {time}, ~{calories} calories")
        '''
    except Exception as e:
        print(f"Error analyzing recipes: {str(e)}")


if __name__ == "__main__":
    main()
