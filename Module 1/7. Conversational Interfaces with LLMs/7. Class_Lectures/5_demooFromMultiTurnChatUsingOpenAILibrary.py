from openai import OpenAI

# 1. Initialize OpenAI client to point at Gemini endpoint
client = OpenAI(
    api_key="",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 2. Initialize conversation history
messages = [
    {"role": "system", "content": "You are a career counselor. Give clear, empathetic advice and ask follow-up questions."},
    {"role": "user", "content": "Hi, I'm Nikhil. I enjoy math and programming and I just completed my B.Sc."}
]

# 3. Send first request
response = client.chat.completions.create(
    model="gemini-3.6-flash",  # Use Gemini model name
    messages=messages
)
assistant_reply = response.choices[0].message.content

print("Assistant:", assistant_reply)

# 4. Append assistant reply and continue the conversation
messages.append({"role": "assistant", "content": assistant_reply})
messages.append({"role": "user", "content": "What career paths should I consider?"})

response = client.chat.completions.create(
    model="gemini-3.6-flash",
    messages=messages
)
assistant_reply = response.choices[0].message.content

print("Assistant:", assistant_reply)

