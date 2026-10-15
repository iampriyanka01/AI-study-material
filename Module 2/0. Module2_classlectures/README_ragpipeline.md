# DilSeVector Concierge — Converted Notebook + Local Script

This folder contains two ways to run the Session 5 "DilSeVector Concierge"
RAG (Retrieval-Augmented Generation) build, converted from
`Session5_Concierge.ipynb`:

| File | Where to run it | What it is |
|---|---|---|
| `Session5_Concierge_clean.ipynb` | Google Colab | The same notebook, cleaned up (large jupyter-widget clutter removed) and re-commented for clarity. Uses Colab Secrets for API keys. |
| `dilsevector_concierge.py` | Local VS Code / terminal | The same logic as a plain Python script, organized into functions with a `main()` that runs every level's demo in order. Uses environment variables / a `.env` file for API keys instead of Colab Secrets. |
| `requirements.txt` | Local only | pip dependencies for the `.py` script. |

## What the notebook/script builds

A chat concierge that answers coffee questions **only** from real content
in a set of local HTML "coffee pages," instead of letting a language model
invent facts. It's built up in levels, each fixing a specific weakness of
the one before it:

| Level | The concierge… | Called |
|---|---|---|
| L0 | …makes things up | (no RAG) |
| L1 | …reads the manual first | **Basic RAG** |
| L2 | …follows a written script | **Prompt template** |
| L3 | …remembers you | **Memory RAG** |
| L4 | …has specialist desks | **Branched RAG** |
| L5 | …spends only when needed | **Adaptive RAG** |
| L6 | …checks its own work | **Corrective RAG** |
| Bonus | …refuses to guess on safety questions | **Safety branch** (the notebook's homework exercise) |

Every function intentionally uses only basic Python (`for`, `if`, lists,
dicts, f-strings) — no comprehensions, no regex, no classes — so the RAG
*logic* stays visible instead of being hidden behind clever syntax.

## What you need before running either version

1. **A Google Gemini API key** — get one free at
   https://aistudio.google.com/app/apikey
2. **A Pinecone API key and an existing index named `coffeeindex`** — sign
   up at https://app.pinecone.io/. The index must be created with
   **dimension 384** and **metric cosine**, to match the
   `paraphrase-MiniLM-L6-v2` embedding model used here. (If your index
   doesn't exist yet, create it in the Pinecone console, or via
   `pc.create_index(name="coffeeindex", dimension=384, metric="cosine", spec=...)`.)
3. **The `coffee_pages/` folder of source HTML files** — download from the
   shared Google Drive folder used in the original course notebook:
   https://drive.google.com/drive/folders/1wlvoSwA5H_JtU_lb0rSzDakPxZanhkmU?usp=sharing
   Place the extracted `coffee_pages` folder next to the notebook/script.

## 1. Running in Google Colab (`Session5_Concierge_clean.ipynb`)

1. Upload `Session5_Concierge_clean.ipynb` to Google Colab (File → Upload
   notebook), or open it from Google Drive.
2. Add your two secrets in Colab's **🔑 Secrets** panel (left sidebar):
   - `GOOGLE_API_KEY`
   - `PINECONE_API_KEY`
   Enable "Notebook access" for both.
3. Upload the `coffee_pages/` folder into the Colab session's working
   directory (Files panel on the left → upload folder, or mount Google
   Drive and adjust the path).
4. Run the cells top to bottom.
   - **Cell 3** ("Put them in Pinecone") only needs to run with
     `UPLOAD = True` the *first* time — after that, leave it `False`.
   - Every other cell (Levels 0–6 and the bonus safety branch) can be
     re-run any time to see that level's demo.

## 2. Running locally in VS Code (`dilsevector_concierge.py`)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows PowerShell/CMD

# 2. Install dependencies
pip install -r requirements.txt
```

Set your API keys as environment variables, or create a `.env` file next
to the script:

```
GOOGLE_API_KEY=your-gemini-key-here
PINECONE_API_KEY=your-pinecone-key-here
```

Place the `coffee_pages/` folder in the same directory as
`dilsevector_concierge.py`, then run:

```bash
python dilsevector_concierge.py
```

On the **first run only**, open the script and set:

```python
UPLOAD_TO_PINECONE = True
```

near the bottom of the file, so the coffee pages actually get embedded and
stored in your Pinecone index. Set it back to `False` afterwards — the
data persists in Pinecone between runs, and re-uploading every time just
wastes API calls.

Running the script executes every level's demo in sequence (L0 through L6,
plus the bonus safety branch) and prints the results to the terminal —
this mirrors running every cell of the notebook top to bottom. If you'd
rather experiment interactively, import the script's functions into a
Python REPL or a Jupyter kernel running locally:

```python
from dilsevector_concierge import setup, search, answer
ctx = setup()
print(answer(ctx["model"], ctx["embedder"], ctx["index"], "Is turmeric coffee caffeine free?"))
```

## What changed from the original notebook

- **Removed ~6,000 lines of cached Jupyter-widget metadata** that had
  accumulated in the `.ipynb`'s `metadata.widgets` section (leftover
  progress-bar state from a previous run) — this doesn't affect behavior,
  only file size and load time.
- **Colab-only secrets (`google.colab.userdata.get(...)`)** → in the
  `.py` script, replaced with a plain environment-variable read
  (`os.getenv(...)`), optionally loaded from a local `.env` file via
  `python-dotenv`. The `.ipynb` keeps the original Colab Secrets approach,
  since it's the natural way to store keys in Colab.
- **Cell-by-cell notebook globals** (`model`, `embedder`, `index`,
  `history`, `preferences`, etc.) → in the `.py` script, `setup()` returns
  these as a dict instead of relying on notebook-style global variables,
  and each level's demo is wrapped in its own named function
  (`level0_demo`, `level1_demo`, ... `safety_branch_demo`), called in order
  from `main()`.
- Every original comment was preserved and extra comments were added
  explaining *why* each function exists and what problem it fixes,
  matching the level-by-level teaching structure of the source notebook.
- No functional logic was changed — same routing rules, same prompt
  template, same evaluation checks, same retry logic.

## Troubleshooting

- **`PineconeValueError: ... keyword-only ...`** — recent `pinecone`
  client versions require keyword arguments (e.g. `index.upsert(vectors=..., namespace=...)`
  instead of positional arguments). Both files here already use the
  keyword form.
- **`RuntimeError: GOOGLE_API_KEY not found`** — in Colab, check the key
  name matches exactly in the Secrets panel and that "Notebook access" is
  toggled on; locally, check your `.env` file is in the same folder you're
  running the script from.
- **Embedding dimension mismatch when upserting to Pinecone** — your
  `coffeeindex` index must be created with `dimension=384` (matching
  `paraphrase-MiniLM-L6-v2`'s output size). If you created it with a
  different dimension, either recreate the index or switch to a matching
  embedding model.
- **`model.generate_content(...)` errors about the model name** — Gemini
  model names/availability can change between accounts and API versions;
  if `gemini-3.6-flash` isn't available on your account, swap in whatever
  current Gemini model name your API key has access to.
