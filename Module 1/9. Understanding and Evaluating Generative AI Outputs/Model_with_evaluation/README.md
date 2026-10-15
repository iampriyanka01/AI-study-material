# Study Buddy Chatbot + Evaluator

A tutor chatbot ("Study Buddy") paired with an automated "LLM-as-judge"
evaluator, built on Google's Gemini API (via the `google-genai` SDK) and
served through a **Streamlit** web UI.

- **Study Buddy** answers school-related questions in a simple, friendly
  tone and gently redirects off-topic questions.
- **Evaluator** is a *second*, independent Gemini instance that grades
  every one of Study Buddy's answers against a fixed rubric (accuracy,
  clarity, coherence, tone, relevance, conciseness) and returns
  structured JSON feedback.
- The Streamlit app shows the evaluator's scores and comments **right
  after every answer**, for every question you ask — not just once.

This is a refactor of an earlier two-script version
(`smartbuddychatbot.py` + `evaluatormodel.py`). Those two files
duplicated the API-key loading, client setup, and the Study Buddy
system prompt. Everything has now been split into small, single-purpose
modules so nothing is defined twice.

## Project structure

```
study_buddy_app/
├── app.py            # Streamlit UI — the main app, run this
├── cli.py            # optional plain-terminal version (same modules, no UI)
├── config.py         # loads .env, creates the shared Gemini client, model names
├── prompts.py         # the two system prompts (Study Buddy + Evaluator), defined once
├── study_buddy.py     # target-model logic: create a chat session, get a reply
├── evaluator.py        # evaluator-model logic: create a session, grade a reply, parse JSON
├── requirements.txt
└── README.md
```

### How the pieces fit together

```
config.py  ──┬─▶ study_buddy.py ──┐
             │                    ├─▶ app.py  (Streamlit UI)
             └─▶ evaluator.py ────┤
                                  └─▶ cli.py   (terminal UI)

prompts.py ──▶ study_buddy.py, evaluator.py
```

- **`config.py`** is the only place that reads `GEMINI_API_KEY` and
  creates a `genai.Client`. Every other module imports `client` from
  here instead of re-creating it.
- **`prompts.py`** is the only place either system prompt is written
  out. Edit a prompt once, and both the UI and the CLI pick up the
  change automatically.
- **`study_buddy.py`** and **`evaluator.py`** each expose two small
  functions: `create_..._chat()` (start a session) and
  `get_..._response()` / `evaluate_response()` (send one message).
  Neither file knows anything about Streamlit.
- **`app.py`** and **`cli.py`** are the only files that know about the
  *user interface*. They call the functions above and display the
  results; they contain no Gemini API calls or prompt text themselves.

If you want to reuse this pipeline somewhere else (a Slack bot, a batch
script, a test suite), you only need `config.py`, `prompts.py`,
`study_buddy.py`, and `evaluator.py` — the UI files are optional.

## Requirements

- Python 3.9+
- A free Gemini API key from https://aistudio.google.com/app/apikey

## Setup

### 1. Create and activate a virtual environment
```bash
python -m venv venv
```
- macOS/Linux: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

In VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `./venv`.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API key
Create a file named `.env` in this folder containing:
```
GEMINI_API_KEY=your_key_here
```
Never commit this file to version control.

### 4. Run the app

**Streamlit UI (recommended):**
```bash
streamlit run app.py
```
This opens a browser tab with a chat box. Type a question, and both
Study Buddy's answer and the evaluator's scored feedback appear
together, for every question you ask.

**Plain terminal version:**
```bash
python cli.py
```
Same pipeline, printed to the console instead of a browser. Type `exit`
to quit.

## Setup — Google Colab

Streamlit doesn't run well inside a Colab cell, so in Colab use `cli.py`
instead:

1. Install dependencies:
   ```python
   !pip install -q google-genai python-dotenv
   ```
2. Set your key using Colab Secrets (🔑 icon in the left sidebar → add a
   secret named `GEMINI_API_KEY`), then load it before running the script:
   ```python
   import os
   from google.colab import userdata
   os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
   ```
   Since `config.py` calls `load_dotenv()` (a no-op if there's no `.env`
   file) and then reads `os.getenv("GEMINI_API_KEY")`, setting the
   environment variable this way in the same session works fine — you
   don't need an actual `.env` file in Colab.
3. Run:
   ```python
   !python cli.py
   ```

## If you get a 404 on the model name

`config.py` currently sets:
```python
STUDY_BUDDY_MODEL = "gemini-3.6-flash"
EVALUATOR_MODEL = "gemini-3.5-flash"
```
Gemini model names and availability change over time and can vary by
API key/region. If you get a `404 NOT_FOUND` error, list the models
your key actually has access to and update those two constants in
`config.py`:

```python
from google import genai
client = genai.Client(api_key="your_key_here")
for m in client.models.list():
    print(m.name)
```

## How the evaluator pipeline works, step by step

1. **`app.py`** (or `cli.py`) calls `study_buddy.create_study_buddy_chat()`
   once at the start of the session, and `evaluator.create_evaluator_chat()`
   once too — two independent Gemini chat sessions.
2. On each question, `study_buddy.get_study_buddy_response(...)` sends
   the question to the **target model** and returns its answer.
3. `evaluator.evaluate_response(...)` sends the *original question* and
   *Study Buddy's answer* to the **evaluator model**, asking it to grade
   the answer against the rubric in `prompts.EVALUATOR_SYSTEM_PROMPT`.
4. `evaluator.safe_parse_json(...)` scans the evaluator's reply for the
   first balanced `{...}` block and parses it as JSON, tolerating minor
   formatting issues (single quotes, trailing commas) in case the model
   doesn't return perfectly strict JSON.
5. **`app.py`** stores both the answer and the evaluation in
   `st.session_state.messages` and renders them together — the answer as
   a normal chat bubble, followed by an "🧪 Evaluator feedback" panel
   with per-dimension scores (out of 5), short comments, and an overall
   summary. This repeats for every question, and the full history stays
   visible as you keep chatting. `cli.py` does the same thing, printed
   as JSON to the terminal.

## Notes

- **API cost**: the Gemini free tier has rate limits (requests per
  minute/day) but no billing requirement — if you hit a `429`, it's a
  temporary rate limit rather than an exhausted balance, so a short wait
  and retry should work.
- **Every question makes 2 API calls** (one for Study Buddy, one for the
  evaluator), so the app uses roughly double the quota of a plain
  chatbot per question.
- If the evaluator's response fails to parse as JSON, the UI shows a
  warning instead of crashing — Study Buddy's answer is still displayed
  normally.
