from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Gemini API key not found. Check your .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)

# -----------------------------
# 2️⃣ Define Real Tool Functions
# -----------------------------
def add(a: int, b: int) -> int:
    print(f"Executing add({a}, {b})")
    return a + b

def sub(a: int, b: int) -> int:
    print(f"Executing sub({a}, {b})")
    return a - b   # ✅ fixed subtraction

def multiply(a: int, b: int) -> int:
    print(f"Executing multiply({a}, {b})")
    return a * b

def division(a: int, b: int) -> float:
    print(f"Executing division({a}, {b})")
    return a / b   # ✅ fixed division

available_functions = {
    "add": add,
    "sub": sub,
    "multiply": multiply,
    "division": division,
}

# -----------------------------
# 3️⃣ Declare Tool Schemas
# -----------------------------
add_decl = types.FunctionDeclaration(
    name="add",
    description="Adds two integers.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "a": types.Schema(type="INTEGER", description="First integer"),
            "b": types.Schema(type="INTEGER", description="Second integer"),
        },
        required=["a", "b"],
    ),
)

sub_decl = types.FunctionDeclaration(
    name="sub",
    description="Subtracts two integers.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "a": types.Schema(type="INTEGER", description="First integer"),
            "b": types.Schema(type="INTEGER", description="Second integer"),
        },
        required=["a", "b"],
    ),
)

mul_decl = types.FunctionDeclaration(
    name="multiply",
    description="Multiplies two integers.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "a": types.Schema(type="INTEGER", description="First integer"),
            "b": types.Schema(type="INTEGER", description="Second integer"),
        },
        required=["a", "b"],
    ),
)

div_decl = types.FunctionDeclaration(
    name="division",
    description="Divides two integers.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "a": types.Schema(type="INTEGER", description="First integer"),
            "b": types.Schema(type="INTEGER", description="Second integer"),
        },
        required=["a", "b"],
    ),
)

tools = [types.Tool(function_declarations=[add_decl, sub_decl, mul_decl, div_decl])]

# -----------------------------
# 4️⃣ Common Config
# -----------------------------
config = types.GenerateContentConfig(tools=tools)

# -----------------------------
# 5️⃣ Prepare Prompt
# -----------------------------
user_prompt = "What is 3 * 12? Also, what is 113 - 49? what is 11 + 26? what is 27 / 3?"
messages = [types.Content(role="user", parts=[types.Part(text=user_prompt)])]

# -----------------------------
# 6️⃣ First Call
# -----------------------------
print(">>> Asking model...")
response_1 = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=messages,
    config=config
)

# -----------------------------
# 7️⃣ Inspect model's response
# -----------------------------
content_1 = response_1.candidates[0].content
print(">>> Model response:")
for part in content_1.parts:
    if part.text:
        print(f"Model text: {part.text}")
    if part.function_call:
        print(f"Function call: {part.function_call.name} with args {part.function_call.args}")
messages.append(content_1)

# -----------------------------
# 8️⃣ Execute any function_calls
# -----------------------------
tool_outputs = []
for part in content_1.parts:
    if part.function_call:
        name = part.function_call.name
        args = part.function_call.args
        print(f"> Function call: {name} with args {args}")
        fn = available_functions.get(name)
        try:
            result = fn(**args)
            tool_outputs.append(
                types.Part.from_function_response(
                    name=name,
                    response={"result": result},
                )
            )
        except Exception as e:
            tool_outputs.append(
                types.Part.from_function_response(
                    name=name,
                    response={"error": str(e)},
                )
            )

# -----------------------------
# 9️⃣ Second Turn
# -----------------------------
if tool_outputs:
    messages.append(types.Content(role="tool", parts=tool_outputs))
    print(">>> Providing function outputs back to model...")
    response_2 = client.models.generate_content(
        model="gemini-3.5-flash",   # ✅ use supported model
        contents=messages,
        config=config
    )
    final = response_2.candidates[0].content
    print("\n Final Answer:")
    print("".join(part.text or "" for part in final.parts))
else:
    print("\nResponse:")
    print("".join(part.text or "" for part in content_1.parts))
