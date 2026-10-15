# RAG Chunking & Retrieval Strategies Demo

This project demonstrates a full RAG (Retrieval-Augmented Generation) pipeline
using a mock "personal loan policy" document, covering two stages:

1. **Chunking Techniques** (splitting documents into pieces)
   - Fixed-size chunking
   - Hierarchical chunking
   - Semantic chunking

2. **Retrieval Strategies** (picking the right chunks for a query)
   - Vanilla Top-K
   - MMR (Maximum Marginal Relevance)
   - Reranking (by recency/year)

## Project Structure

```
rag_chunking_demo/
├── chunking_demo.py     # Main script
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Prerequisites

- Python 3.9+ installed
- VS Code with the Python extension installed
- Internet connection (needed the first time you run the script, to
  download the `all-MiniLM-L6-v2` sentence-transformer model and the
  NLTK `punkt_tab` tokenizer data)

## Step-by-Step Setup

### 1. Open the project in VS Code

Open the `rag_chunking_demo` folder in VS Code (`File > Open Folder...`).

### 2. Create a virtual environment

Open a terminal inside VS Code (`` Ctrl+` `` / `` Cmd+` ``) and run:

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

After activation, you should see `(venv)` at the start of your terminal prompt.

> If PowerShell blocks the activation script with an execution-policy error,
> run this once, then retry activation:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. Select the virtual environment as your VS Code interpreter

- Press `Ctrl+Shift+P` (`Cmd+Shift+P` on Mac) to open the Command Palette
- Type and select **"Python: Select Interpreter"**
- Choose the interpreter located inside the `venv` folder
  (e.g. `./venv/bin/python` or `.\venv\Scripts\python.exe`)

### 4. Install the dependencies

With the virtual environment activated, run:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `nltk` – sentence tokenization
- `sentence-transformers` – embedding model (`all-MiniLM-L6-v2`)
- `scikit-learn` – cosine similarity
- `numpy` – array operations

> **Note:** `sentence-transformers` pulls in PyTorch, so this install can
> take a few minutes and needs a few GB of disk space.

### 5. Run the script

```bash
python chunking_demo.py
```

Or in VS Code: open `chunking_demo.py` and click the **Run** (▶) button in
the top-right corner, or press `F5` / `Ctrl+F5` (Run Without Debugging).

The first run will also download:
- The `all-MiniLM-L6-v2` model (~80 MB) from Hugging Face
- The NLTK `punkt_tab` tokenizer data

Both are cached locally afterward, so subsequent runs are much faster.

### 6. What you should see

The script prints, in order:
1. **Fixed-Size Chunking** results (word-count based chunks with overlap)
2. **Hierarchical Chunking** results (split by "Section N:" headers)
3. **Semantic Chunking** results (sentences grouped by embedding similarity)
4. **Vanilla Top-K** retrieval results with similarity scores
5. **MMR** retrieval results (diversity-aware)
6. **Reranking by recency** results (same chunks as vanilla top-3, reordered by year)

## Deactivating the virtual environment

When you're done, simply run:

```bash
deactivate
```

## Troubleshooting

- **`ModuleNotFoundError`**: Make sure the virtual environment is activated
  (you should see `(venv)` in your terminal) and that you ran
  `pip install -r requirements.txt` inside it.
- **Slow first run / model download hangs**: Check your internet connection —
  `sentence-transformers` needs to download the model from Hugging Face the
  first time it's used.
- **`nltk` tokenizer errors**: The script calls `nltk.download("punkt_tab")`
  automatically, but if it fails (e.g. due to a firewall), run this manually
  in a Python shell inside your activated venv:
  ```python
  import nltk
  nltk.download("punkt_tab")
  ```
