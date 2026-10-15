# DilSeVector Concierge — RAG on LangChain (local / VS Code version)

This is the local-machine version of `RAG_LangChain_Colab.ipynb`. It's the
same 14-part walkthrough — rebuilding the hand-written Session 5 Concierge
on top of LangChain — as one Python script (`rag_langchain_pipeline.py`) you
can run from VS Code's terminal instead of a Colab cell.

## What it does

| Part | Topic |
|---|---|
| 1 | The one idea: everything is a Runnable |
| 2 | The pipe `\|` and LCEL |
| 3 | Models and messages (`init_chat_model`) |
| 4 | Prompts: system vs human, few-shot |
| 5 | Output parsers, including JSON |
| 6 | Documents, loaders, splitters |
| 7 | Embeddings and vector stores (hybrid search) |
| 8 | Retrievers and the alpha knob (dense vs sparse) |
| 9 | Smarter retrieval: multi-query, self-query, compression, diversity, budget |
| 10 | Rebuilding the Concierge as one LangChain chain |
| 11 | Conversation memory (manual + managed) |
| 12 | Self-Adaptive RAG (plan → answer → verify → retry) |
| 13 | Seeing inside your chain (debugging) |

**Important:** LangChain has search, prompting, routing, reranking, and
memory components — but **no tool that checks whether an answer is actually
true**. The four citation/coverage checks from your Session 5 notebook
(`find_citations`, `check_citations_real`, etc.) have no LangChain
replacement and are worth carrying into this pipeline yourself; Part 10's
`has_valid_citations()` is a starting point.

Every function mirrors the original notebook's cells closely — read them
side by side and you'll recognize every line.

## 1. Prerequisites

- **Python 3.9–3.11** (`langchain-pinecone`'s current release does not
  support Python 3.12, 3.13, or 3.14 — see Troubleshooting below)
- A Google AI Studio API key (for Gemini + embeddings):
  https://aistudio.google.com/app/apikey
- A Pinecone account + API key: https://app.pinecone.io/
- A Pinecone index named **`coffee-hybrid`**, created for **hybrid search**
  (metric = `dotproduct`), with dimension matching Google's
  `text-embedding-004` model (768)
- The `coffee_pages/` folder of source HTML files, placed next to
  `rag_langchain_pipeline.py`

## 2. Set up your environment

```bash
# Create and activate a virtual environment (use a 3.9-3.11 interpreter)
python3.11 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Add your API keys

Create a file named `.env` in this same folder with:

```
GOOGLE_API_KEY=your-gemini-api-key-here
PINECONE_API_KEY=your-pinecone-api-key-here
```

**Never commit this file to git.**

## 4. Add your source data

Place your `coffee_pages/` folder next to `rag_langchain_pipeline.py`:

```
your-project/
├── rag_langchain_pipeline.py
├── requirements.txt
├── README.md
├── .env
└── coffee_pages/
    ├── page1.html
    ├── page2.html
    └── ...
```

## 5. Run it

**First run only** — embed and upload your pages into the Pinecone hybrid
index:

```bash
python rag_langchain_pipeline.py --upload
```

**Every run after that** — the data is already indexed, so just:

```bash
python rag_langchain_pipeline.py
```

This walks through Parts 1–13 in order, printing each part's output to the
terminal, exactly like running the notebook top to bottom.

### Running only some parts

Some parts (9, 11, 12) make several LLM calls and take longer. Run a subset
with `--parts`:

```bash
# Just the framework basics, no API calls
python rag_langchain_pipeline.py --parts 1,2

# Jump straight to the rebuilt Concierge
python rag_langchain_pipeline.py --parts 10

# The adaptive RAG demo (also runs the compression setup it depends on)
python rag_langchain_pipeline.py --parts 12
```

Parts that build shared objects other parts depend on (the model, the
embeddings, the loaded pages, the retriever, and the concierge prompt/chain)
are always set up regardless of which `--parts` you pass, so any part can
run on its own.

## 6. Where to look in the code

`rag_langchain_pipeline.py` is organized into clearly labeled banner
sections, one per Part, in the same order as the notebook:

- `SECTION 0` — loads your `.env` keys, defines shared module-level objects
- `PART 1` – `PART 2` — the core LangChain idea (Runnables, LCEL), no API key needed
- `PART 3` — builds the shared `llm` object
- `PART 4` – `PART 5` — prompts and output parsers
- `PART 6` — `read_one_page()` / `load_coffee_pages()` — turns your HTML
  files into LangChain `Document` objects, same chunking logic as Session 5
- `PART 7` — builds `embeddings` and the hybrid `retriever` (and uploads to
  Pinecone if you pass `--upload`)
- `PART 8` — the alpha knob demo (dense vs sparse search)
- `PART 9.1`–`9.6` — one function per smarter-retrieval technique
- `PART 10` — `format_context`, `answer_question`, `concierge_chain`,
  `full_chain`, streaming, and citation checking/repair
- `PART 11.1`/`11.2` — manual memory (`chat()`) and managed memory
  (`RunnableWithMessageHistory`)
- `PART 12` — `plan_query`, `answer_once`, `adaptive_answer` (verify + retry)
- `PART 13` — debugging: inspecting retrieved documents and the exact prompt
- `MAIN` — the `--parts`/`--upload` driven runner at the bottom

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'langchain.retrievers'` | pre-v1 import somewhere | this script already uses the v1 paths (`langchain_classic...`) |
| `AttributeError: ... has no attribute 'get_relevant_documents'` | removed in v1 | this script already uses `.invoke()` |
| `pip install` fails on `langchain-pinecone` | Python 3.12+ isn't supported yet by its pinned `simsimd` dependency | use Python 3.9–3.11 |
| Pinecone dimension mismatch | index wasn't created with the `text-embedding-004` dimension (768) | recreate the `coffee-hybrid` index with the matching dimension and `dotproduct` metric |
| `SelfQueryRetriever` fails on import | `lark` missing | confirm `lark` installed (`pip install -r requirements.txt`) |
| Self-query returns zero documents | the model invented a filter value that doesn't exist | the script already falls back to plain retrieval in this case |
| Citations vanished after compression | `LLMChainExtractor` rewrote `page_content` | ids are kept safe in `metadata`, not `page_content` |
| `KeyError` on a literal `{` in a prompt | single braces are template slots | double them: `{{` and `}}` (already done in the router/verifier prompts) |
| Answers seem off after running Part 8 or 9 | `retriever.top_k`/`retriever.alpha` left at a demo's value | the script resets these after each demo, but re-run with `--parts` to isolate |
| First run is very slow | cold start, embedding + BM25 fitting all your pages | expected once per process; subsequent `.invoke()` calls are fast |

## Appendix: migrating even older LangChain code

If you find older blog-post code using `langchain.retrievers`,
`langchain.chains.LLMChain`, or `.get_relevant_documents()`, see the
Appendix A / Appendix B markdown cells at the end of `RAG_LangChain_Colab.ipynb`
for the full old → new mapping. This script already uses the current (v1)
import paths and methods throughout.
