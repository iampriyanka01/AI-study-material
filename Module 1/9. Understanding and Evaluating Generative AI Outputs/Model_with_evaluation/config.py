"""
config.py
=========
Single source of truth for app-wide configuration.

Every other module (study_buddy.py, evaluator.py, app.py, cli.py) imports
`client`, `STUDY_BUDDY_MODEL`, and `EVALUATOR_MODEL` from HERE instead of
loading the .env file or creating a genai.Client() themselves. This is
what removes the duplicated "load key / create client" boilerplate that
used to be copy-pasted at the top of both smartbuddychatbot.py and
evaluatormodel.py.
"""

import os
from dotenv import load_dotenv
from google import genai

# ---------------------------------------------------------------------
# 1) Load GEMINI_API_KEY from a local .env file (see README.md).
#    load_dotenv() is a harmless no-op if no .env file exists (e.g. in
#    Colab, where you set the environment variable directly).
# ---------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "No GEMINI_API_KEY found. Create a '.env' file next to this app "
        "containing: GEMINI_API_KEY=your_key_here"
    )

# ---------------------------------------------------------------------
# 2) One shared Gemini client for the whole app.
# ---------------------------------------------------------------------
client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------
# 3) Model names, kept in one place so they're easy to swap.
#    If you get a 404 NOT_FOUND error, list the models your API key
#    currently has access to and update these two constants:
#
#       from google import genai
#       client = genai.Client(api_key="your_key_here")
#       for m in client.models.list():
#           print(m.name)
# ---------------------------------------------------------------------
STUDY_BUDDY_MODEL = "gemini-3.5-flash"   # the "target" model students talk to
EVALUATOR_MODEL = "gemini-3.5-flash"     # the "judge" model that grades it
