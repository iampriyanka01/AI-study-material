"""
dilsevector_concierge.py

Session 5 - The DilSeVector Concierge
======================================
Converted from: Session5_Concierge.ipynb

WHAT THIS SCRIPT IS
--------------------
A step-by-step build of a Retrieval-Augmented Generation (RAG) chat
concierge that answers coffee questions using only real content pulled
from a small set of local HTML "coffee pages" (via Pinecone search),
instead of letting a language model invent facts.

The notebook (and this script) build the concierge up in *levels*, each
one fixing a specific weakness of the level before it:

    L0  makes things up                 (no RAG at all)
    L1  reads the manual first          Basic RAG
    L2  follows a written script        Prompt template
    L3  remembers you                   Memory RAG (history + preferences)
    L4  has specialist desks            Branched RAG (route by question type)
    L5  spends only when needed         Adaptive RAG (cheap vs careful search)
    L6  checks its own work             Corrective RAG (verify before answering)
    +   a bonus "safety" branch added as the homework exercise in the
        original notebook (answers safety-sensitive questions only from
        the "Benefits, Cautions & Practical Tips" section).

Every function below intentionally uses only basic Python (for loops, if
statements, lists, dictionaries, f-strings) - no comprehensions, no
regular expressions, no classes - so the RAG *logic* stays easy to follow
rather than being hidden behind clever syntax. This mirrors the teaching
style of the original notebook.

WHAT YOU NEED BEFORE RUNNING THIS
-----------------------------------
1. A Google Gemini API key (https://aistudio.google.com/app/apikey)
2. A Pinecone API key (https://app.pinecone.io/) and an existing Pinecone
   index named "coffeeindex" (see README.md for how to create one that
   matches the embedding model used here).
3. The "coffee_pages" folder of HTML files used as source data. In the
   original notebook these were downloaded from a shared Google Drive
   folder - see README.md for the link and expected folder layout.

HOW TO RUN (from VS Code / terminal)
-------------------------------------
1. Create and activate a virtual environment (recommended):
       python -m venv .venv
       .venv\\Scripts\\activate      (Windows)
       source .venv/bin/activate    (macOS/Linux)

2. Install dependencies:
       pip install -r requirements.txt

3. Set your API keys as environment variables (or in a local .env file -
   see README.md):
       GOOGLE_API_KEY=your-gemini-key
       PINECONE_API_KEY=your-pinecone-key

4. Place the "coffee_pages" folder (containing the .html source pages)
   in the same directory as this script.

5. Run the script:
       python dilsevector_concierge.py

   The first run should be done with UPLOAD_TO_PINECONE = True (below)
   so the coffee pages get embedded and stored in Pinecone. After that,
   set it back to False - the data is already there, so there is no need
   to re-upload it every run (and doing so wastes API calls).
"""

import os
import glob
import time

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai
from dotenv import load_dotenv          # lets us keep API keys in a local .env file

load_dotenv()  # loads GOOGLE_API_KEY / PINECONE_API_KEY from a ".env" file, if present


# ---------------------------------------------------------------------------
# CELL 1 — SETUP
# ---------------------------------------------------------------------------
# Connects to Gemini (the language model) and Pinecone (the vector search
# database), and loads the sentence-embedding model used to turn text into
# vectors. Run this once at the start of every session.
# ---------------------------------------------------------------------------

# Folder containing the source coffee-page HTML files (see README.md for
# where to download these from). Change this if you keep the folder
# somewhere else.
COFFEE_PAGES_FOLDER = "coffee_pages"

# Name of the existing Pinecone index that stores the embedded coffee pages.
PINECONE_INDEX_NAME = "coffeeindex"

# Pinecone "namespace" - a logical partition inside the index, similar to a
# table/collection. All coffee-page pieces live in this one namespace.
NAMESPACE = "docs"


def setup():
    """
    Connect to Gemini + Pinecone and load the embedding model.

    Returns a dict with the three things every other function in this
    script needs: {"model": ..., "embedder": ..., "index": ...}
    """

    gemini_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Set it as an environment variable, "
            "or add GOOGLE_API_KEY=... to a .env file next to this script."
        )

    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        raise RuntimeError(
            "PINECONE_API_KEY not found. Set it as an environment variable, "
            "or add PINECONE_API_KEY=... to a .env file next to this script."
        )

    # Configure the Gemini client and pick the chat model we'll use to
    # generate answers.
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-3.6-flash")

    # This embedding model turns text into a 384-dimensional vector. It must
    # match whatever dimension your Pinecone index ("coffeeindex") was
    # created with - see README.md.
    embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    # Connect to Pinecone and get a handle to the existing index.
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(PINECONE_INDEX_NAME)

    print("Ready. Embedding size:", embedder.get_sentence_embedding_dimension())

    return {"model": model, "embedder": embedder, "index": index}


# ---------------------------------------------------------------------------
# CELL 2 — READ THE COFFEE PAGES
# ---------------------------------------------------------------------------
# Each coffee page has the same 8 sections (Overview, Ingredients, Method,
# and so on). We split every page into ONE CHUNK PER SECTION, which gives
# small, tidy pieces to search over instead of one huge blob per page.
# ---------------------------------------------------------------------------

def read_one_page(filename):
    """Turn one HTML file into a list of section pieces (chunks)."""

    html = open(filename, encoding="utf-8", errors="ignore").read()
    soup = BeautifulSoup(html, "html.parser")

    # Strip out anything that isn't real page content before extracting text.
    for tag in soup(["script", "style", "nav", "footer", "head"]):
        tag.decompose()

    title = soup.find("h1").get_text(strip=True)
    page_id = os.path.basename(filename).replace(".html", "")

    pieces = []
    for heading in soup.find_all("h2"):  # each <h2> starts a new section
        section_name = heading.get_text(strip=True)
        section_text = ""

        # Walk forward through the HTML from this heading until the NEXT
        # <h2>, collecting all the text in between as this section's content.
        for item in heading.next_siblings:
            if getattr(item, "name", None) == "h2":
                break
            if hasattr(item, "get_text"):
                section_text = section_text + " " + item.get_text(" ", strip=True)
        section_text = " ".join(section_text.split())  # collapse extra whitespace

        # Skip tiny/empty sections - not worth storing as a searchable piece.
        if len(section_text) > 40:
            piece = {}
            piece["id"] = page_id + "__" + section_name[:12]
            piece["page"] = page_id
            piece["title"] = title
            piece["section"] = section_name
            piece["text"] = title + " - " + section_name + ". " + section_text
            pieces.append(piece)

    return pieces


def read_all_pages(folder=COFFEE_PAGES_FOLDER):
    """Read every .html file in `folder` and return one big list of pieces."""

    all_pieces = []
    for filename in sorted(glob.glob(os.path.join(folder, "*.html"))):
        for piece in read_one_page(filename):
            all_pieces.append(piece)

    print("Total pieces:", len(all_pieces))
    if all_pieces:
        print("Example:", all_pieces[0]["id"], "|", all_pieces[0]["section"])
    return all_pieces


# ---------------------------------------------------------------------------
# CELL 3 — PUT THEM IN PINECONE
# ---------------------------------------------------------------------------
# Embed each piece and upsert it into Pinecone with its metadata attached.
# Run this ONCE (set UPLOAD_TO_PINECONE = True the first time), then leave
# it False - the data stays in Pinecone between runs.
# ---------------------------------------------------------------------------

def upload_pieces_to_pinecone(embedder, index, all_pieces, batch_size=50):
    """Embed every piece and upsert it into Pinecone, 50 records at a time."""

    rows = []
    for piece in all_pieces:
        vector = embedder.encode(piece["text"]).tolist()
        info = {
            "page": piece["page"], "title": piece["title"],
            "section": piece["section"], "text": piece["text"],
        }
        rows.append((piece["id"], vector, info))

    start = 0
    while start < len(rows):
        index.upsert(vectors=rows[start:start + batch_size], namespace=NAMESPACE)
        start = start + batch_size

    time.sleep(5)  # give Pinecone a moment to finish indexing
    print("Uploaded", len(rows), "pieces")


# ---------------------------------------------------------------------------
# LEVEL 0 — THE INTERN WHO MAKES THINGS UP (no RAG)
# ---------------------------------------------------------------------------
# Asking the model directly, with no retrieval step at all. This is the
# baseline failure mode: the model will often confidently invent cafes,
# addresses, and facts that don't exist, because it has no real source of
# truth to draw from.
# ---------------------------------------------------------------------------

def level0_demo(model):
    """Ask the model a question with NO retrieval - shows it can hallucinate."""

    question = "Which cafes in Mumbai serve ashwagandha coffee? Give me names and addresses."
    reply = model.generate_content(question)
    print(reply.text)


# ---------------------------------------------------------------------------
# LEVEL 1 — BASIC RAG
# ---------------------------------------------------------------------------
# RAG is just four steps in a row:
#   1. turn the question into a vector
#   2. find the closest pieces (search)
#   3. paste those pieces into a prompt (format_context / build_prompt1)
#   4. ask the model (ask_model)
# ---------------------------------------------------------------------------

def search(embedder, index, question, how_many=4):
    """Step 1 and 2: turn the question into a vector, then find the closest pieces."""

    question_vector = embedder.encode(question).tolist()
    result = index.query(
        vector=question_vector, top_k=how_many,
        namespace=NAMESPACE, include_metadata=True,
    )

    hits = []
    for match in result["matches"]:
        hit = {}
        hit["id"] = match["id"]
        hit["score"] = round(match["score"], 3)
        hit["title"] = match["metadata"]["title"]
        hit["section"] = match["metadata"]["section"]
        hit["text"] = match["metadata"]["text"]
        hits.append(hit)
    return hits


def format_context(hits):
    """Step 3: turn the retrieved pieces into text we can paste into a prompt."""

    lines = []
    for hit in hits:
        lines.append("[" + hit["id"] + "] " + hit["text"])
    return "\n\n".join(lines)


def build_prompt1(question, hits):
    """Step 4 (v1): build a one-off prompt combining the question and retrieved context.

    This is the "quick and dirty" version before we turn it into a reusable
    template in Level 2 below.
    """

    prompt = f"""Answer the question using ONLY the context below.
If the context does not contain the answer, say you do not know.

CONTEXT:
{format_context(hits)}

QUESTION: {question}
ANSWER:"""
    return prompt


def ask_model(model, prompt):
    """Step 4: send the finished prompt to the language model and return its reply text."""
    reply = model.generate_content(prompt)
    return reply.text


def level1_demo(model, embedder, index):
    """Run the four basic RAG steps end-to-end on one example question."""

    question = "What are the ingredients of ashwagandha coffee?"
    hits = search(embedder, index, question)

    print("Found these pieces:")
    for hit in hits:
        print("  ", hit["score"], hit["id"])

    prompt = build_prompt1(question, hits)
    print("\nANSWER:")
    print(ask_model(model, prompt))


# ---------------------------------------------------------------------------
# LEVEL 2 — WRITE THE PROMPT DOWN ONCE (prompt template)
# ---------------------------------------------------------------------------
# In Level 1 the prompt was assembled inline, buried in the middle of the
# code. Here we pull it out into ONE reusable TEMPLATE string with named
# slots ({memory}, {history}, {context}, {question}).
#
# The single most important line in the template is the instruction to say
# "I don't know" when the context doesn't answer the question - without
# that line, the model will quietly go back to making things up (Level 0
# behavior), even though it now has real context available.
# ---------------------------------------------------------------------------

TEMPLATE = """You are the DilSeVector Concierge, a friendly coffee assistant.

RULES:
- Answer ONLY from the CONTEXT below.
- Put the [id] at the end of every sentence that states a fact.
- If the CONTEXT does not answer the question, say you do not know. Never guess.
- Keep it under 80 words.

THINGS WE KNOW ABOUT THIS USER:
{memory}

CONVERSATION SO FAR:
{history}

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""


def build_prompt(question, hits, memory_text="none", history_text="none"):
    """Fill in the shared TEMPLATE with this turn's question, context, memory and history."""
    return TEMPLATE.format(
        memory=memory_text, history=history_text,
        context=format_context(hits), question=question,
    )


def answer(model, embedder, index, question, how_many=4):
    """Convenience wrapper: search, build the templated prompt, and ask the model."""
    hits = search(embedder, index, question, how_many)
    prompt = build_prompt(question, hits)
    return ask_model(model, prompt)


def level2_demo(model, embedder, index):
    """Run one question through the new templated prompt."""
    print(answer(model, embedder, index, "Is turmeric coffee caffeine free?"))


# ---------------------------------------------------------------------------
# LEVEL 3 — MEMORY, AND THE WORD "IT"
# ---------------------------------------------------------------------------
# Problem: if someone asks "Tell me about ashwagandha coffee." and then
# "Compare IT with turmeric coffee.", a plain search on the second question
# alone finds nothing useful - the word "it" carries no search signal on
# its own. The fix is to glue the previous question onto the new one before
# searching (see chat() below).
#
# We track two different kinds of memory, kept in SEPARATE labelled slots so
# they never get mixed into the retrieved CONTEXT:
#   - history     = the chat so far (a running list of turns)
#   - preferences = durable facts about the person ("I prefer low caffeine")
# ---------------------------------------------------------------------------

history = []       # list of ["User", "..."] / ["Assistant", "..."] pairs
preferences = []    # list of sentences describing user preferences


def level3_search_comparison_demo(embedder, index):
    """Show why gluing the previous question onto a follow-up question matters."""

    print("Searching for just: 'Compare it with turmeric coffee'")
    for hit in search(embedder, index, "Compare it with turmeric coffee", 3):
        print("  ", hit.get("page", hit["id"]))

    print("\nSearching with the earlier question glued on:")
    better = "Tell me about ashwagandha coffee. Compare it with turmeric coffee"
    for hit in search(embedder, index, better, 3):
        print("  ", hit["id"])


def find_preferences(text):
    """Pick out sentences where the user states a preference (I prefer / I avoid / I like)."""

    facts = []
    sentences = text.replace("!", ".").replace("?", ".").split(".")
    for sentence in sentences:
        clean = sentence.strip()
        lower = clean.lower()
        if lower.startswith("i prefer") or lower.startswith("i avoid") or lower.startswith("i like"):
            facts.append(clean)
    return facts


def format_history():
    """Turn the last few turns of `history` into text for the prompt's {history} slot."""

    if len(history) == 0:
        return "none"
    lines = []
    for turn in history[-4:]:   # only the last 4 entries (2 user/assistant pairs)
        lines.append(turn[0] + ": " + turn[1])
    return "\n".join(lines)


def format_preferences():
    """Turn the stored `preferences` list into text for the prompt's {memory} slot."""

    if len(preferences) == 0:
        return "none"
    lines = []
    for fact in preferences:
        lines.append("- " + fact)
    return "\n".join(lines)


def chat(model, embedder, index, user_text):
    """
    One full turn of conversation. Same five steps every time:
      1. save any preference the user just stated
      2. search (gluing on the previous question so words like "it" still work)
      3. build the templated prompt (with memory + history filled in)
      4. ask the model
      5. remember this turn for next time
    """

    # 1. remember any preference the user just told us
    for fact in find_preferences(user_text):
        if fact not in preferences:
            preferences.append(fact)

    # 2. search (glue on the last question so words like "it" still resolve)
    search_text = user_text
    if len(history) >= 2:
        search_text = history[-2][1] + " " + user_text

    hits = search(embedder, index, search_text, 4)

    # 3 and 4. build the prompt and ask
    prompt = build_prompt(user_text, hits, format_preferences(), format_history())
    reply = ask_model(model, prompt)

    # 5. remember this turn
    history.append(["User", user_text])
    history.append(["Assistant", reply])

    print("  (used pieces:", end=" ")
    for hit in hits:
        print(hit["id"][:28], end=" ")
    print(")")
    return reply


def level3_demo(model, embedder, index):
    """Run a short multi-turn conversation to show memory + the "it" fix in action."""

    questions = [
        "Tell me about ashwagandha coffee.",
        "I prefer low caffeine drinks.",
        "Compare it with turmeric coffee.",
    ]
    for q in questions:
        print("\nQuestion:", q)
        print("Answer: ", chat(model, embedder, index, q))


# ---------------------------------------------------------------------------
# LEVEL 4 — SPECIALIST DESKS (BRANCHED RAG)
# ---------------------------------------------------------------------------
# Different kinds of questions need different treatment:
#   - a COMPARISON needs pieces about BOTH drinks -> ask for more pieces
#   - a RECOMMENDATION needs the user's preferences -> turn memory on
#   - a LOOKUP wants one exact number -> fewer pieces is actually better
#   - anything else is a plain EXPLAIN
#
# So: look at the question first (route), then decide how to search and
# what to tell the model (settings_for), before actually answering.
# ---------------------------------------------------------------------------

def route(question):
    """Classify a question into one of: compare / recommend / lookup / explain."""

    q = question.lower()
    if "compare" in q or " vs " in q or "difference" in q:
        return "compare"
    if "for me" in q or "suggest" in q or "recommend" in q:
        return "recommend"
    if "how much" in q or "ratio" in q or "how long" in q or "how many" in q:
        return "lookup"
    return "explain"


def settings_for(branch):
    """Return the search/prompt settings that fit a given branch."""

    if branch == "compare":
        return {"how_many": 8, "use_memory": False,
                "extra": "Talk about BOTH drinks. Do not answer about only one."}
    if branch == "recommend":
        return {"how_many": 6, "use_memory": True,
                "extra": "Follow the user's stated preferences and say why."}
    if branch == "lookup":
        return {"how_many": 3, "use_memory": False,
                "extra": "Give the exact number and nothing else."}
    return {"how_many": 5, "use_memory": False, "extra": "Explain in three sentences."}


def level4_routing_demo():
    """Show how different questions get routed to different branches/settings."""

    for q in ["What is cold brew", "Compare ashwagandha with turmeric",
              "Suggest one for me", "What is the cold brew ratio"]:
        branch = route(q)
        settings = settings_for(branch)
        print(branch.ljust(10), "pieces:", settings["how_many"],
              " memory:", settings["use_memory"], " <-", q)


def chat_with_branch(model, embedder, index, user_text):
    """Same idea as chat(), but the branch (from route()) decides the search settings."""

    branch = route(user_text)
    settings = settings_for(branch)

    for fact in find_preferences(user_text):
        if fact not in preferences:
            preferences.append(fact)

    hits = search(embedder, index, user_text, settings["how_many"])

    memory_text = "none"
    if settings["use_memory"]:
        memory_text = format_preferences()

    prompt = build_prompt(user_text, hits, memory_text, format_history())
    prompt = prompt + "\n" + settings["extra"]
    reply = ask_model(model, prompt)

    history.append(["User", user_text])
    history.append(["Assistant", reply])
    print("  (branch:", branch, "| pieces:", len(hits), ")")
    return reply


def level4_demo(model, embedder, index):
    """Run a short conversation through the branch-aware chat function."""

    questions = [
        "Tell me about ashwagandha coffee.",
        "I prefer low caffeine drinks.",
        "Compare it with turmeric coffee.",
        "Compare ashwagandha coffee with turmeric coffee",
    ]
    for q in questions:
        print("\nQuestion:", q)
        print("Answer: ", chat_with_branch(model, embedder, index, q))


# ---------------------------------------------------------------------------
# LEVEL 5 — SPEND ONLY WHEN NEEDED (ADAPTIVE RAG)
# ---------------------------------------------------------------------------
# Some questions are easy, some are hard. Running the expensive settings
# (more retrieved pieces, more tokens) on every question wastes money and
# time. Two clues suggest a question is hard:
#   1. certain words in the question itself ("why", "compare", "how does")
#   2. FLAT search scores - if the best-scoring piece isn't much better than
#      the worst, nothing really matched well, so the search is guessing.
# ---------------------------------------------------------------------------

def looks_hard(question, hits):
    """Decide whether a question needs the more careful (expensive) settings."""

    q = question.lower()

    if "why" in q or "compare" in q or "how does" in q:
        return True

    if len(hits) < 2:
        return True

    best = hits[0]["score"]
    worst = hits[-1]["score"]
    if best - worst < 0.05:      # flat scores: nothing stood out from the rest
        return True

    return False


def choose_settings(embedder, index, question):
    """Peek at the top results first, then decide: 'quick' or 'careful' settings."""

    first_look = search(embedder, index, question, 8)
    if looks_hard(question, first_look):
        return "careful", 12
    return "quick", 4


def level5_demo(embedder, index):
    """Show which questions get routed to 'quick' vs 'careful' settings."""

    for q in ["What is the cold brew ratio",
              "Compare ashwagandha with turmeric coffee",
              "Why is turmeric coffee caffeine free"]:
        style, how_many = choose_settings(embedder, index, q)
        print(style.ljust(9), "pieces:", how_many, " <-", q)


# ---------------------------------------------------------------------------
# LEVEL 6 — CHECK THE ANSWER (CORRECTIVE RAG)
# ---------------------------------------------------------------------------
# Everything up to now tries to WRITE a good answer. Nothing has checked
# whether the answer actually IS good. Four small checks, each catching a
# different failure mode:
#
#   E  Evidence         - did the search actually find anything decent?
#   F  Fair citations   - do the [ids] cited in the answer really exist?
#   C  Coverage         - did the answer address every topic asked about?
#   R  Rules            - does every sentence carry an [id] citation?
# ---------------------------------------------------------------------------

EVIDENCE_MIN = 0.30  # minimum similarity score to count a retrieved piece as "good evidence"


def find_citations(answer_text):
    """Pull out every [id] written in the answer text. No regex - just string splitting."""

    found = []
    pieces = answer_text.split("[")
    for piece in pieces:
        if "]" in piece:
            name = piece.split("]")[0]
            if name not in found:
                found.append(name)
    return found


def check_evidence(hits):
    """E: score how much of the retrieved evidence actually looks relevant."""

    good = 0
    for hit in hits:
        if hit["score"] >= EVIDENCE_MIN:
            good = good + 1
    if good >= 2:
        return 1.0
    if good == 1:
        return 0.5
    return 0.0


def check_citations_real(answer_text, hits):
    """F: check that every [id] cited in the answer actually corresponds to a retrieved piece."""

    real_ids = []
    for hit in hits:
        real_ids.append(hit["id"])

    cited = find_citations(answer_text)
    if len(cited) == 0:
        return 0.0

    good = 0
    for name in cited:
        if name in real_ids:
            good = good + 1
    return good / len(cited)


def check_coverage(answer_text, topics):
    """C: check what fraction of the expected `topics` actually appear in the answer."""

    if len(topics) == 0:
        return 1.0
    found = 0
    for topic in topics:
        if topic.lower() in answer_text.lower():
            found = found + 1
    return found / len(topics)


def check_every_sentence_cited(answer_text):
    """R: check that every non-trivial sentence in the answer carries an [id] citation."""

    for sentence in answer_text.split("."):
        if len(sentence.strip()) < 5:
            continue
        if "[" not in sentence:
            return False
    return True


def check_answer(answer_text, hits, topics):
    """Run all four checks and decide whether the answer passes overall."""

    result = {}
    result["E"] = check_evidence(hits)
    result["F"] = check_citations_real(answer_text, hits)
    result["C"] = check_coverage(answer_text, topics)
    result["R"] = check_every_sentence_cited(answer_text)
    result["passed"] = (
        result["E"] >= 0.7 and result["F"] >= 0.8
        and result["C"] >= 0.7 and result["R"] is True
    )
    return result


def level6_bad_answers_demo(embedder, index):
    """Run four hand-written answers (one good, three broken) through check_answer()."""

    hits = search(embedder, index, "Compare ashwagandha coffee with turmeric coffee", 5)
    topics = ["ashwagandha", "turmeric"]

    first_id = hits[0]["id"]
    second_id = hits[1]["id"]

    good_answer = f"Ashwagandha coffee has caffeine [{first_id}]. Turmeric coffee is caffeine free [{second_id}]."
    no_citation = f"Ashwagandha coffee has caffeine [{first_id}]. Turmeric coffee is caffeine free [{second_id}]. It also cures insomnia."
    fake_page = f"Ashwagandha coffee has caffeine [99_not_a_real_page]. Turmeric coffee is caffeine free [{second_id}]."
    only_half = f"Ashwagandha coffee has caffeine [{first_id}]."

    drafts = [
        ["GOOD", good_answer], ["no citation", no_citation],
        ["fake page", fake_page], ["only half", only_half],
    ]

    for name, draft in drafts:
        print(name.ljust(14), check_answer(draft, hits, topics))


def careful_answer(model, embedder, index, question, topics):
    """
    If the first answer fails the checks, search WIDER and try again - up to
    3 tries total. A repair loop with no limit is just an infinite loop, so
    we give up gracefully and say so honestly after the third attempt.
    """

    how_many = 4
    tries = 0

    while tries < 3:
        hits = search(embedder, index, question, how_many)
        prompt = build_prompt(question, hits)
        if tries > 0:
            prompt = prompt + "\nPut an [id] on EVERY sentence and cover EVERY part of the question."

        draft = ask_model(model, prompt)
        result = check_answer(draft, hits, topics)
        print("  try", tries, "| pieces:", how_many, "|", result)

        if result["passed"]:
            return draft

        how_many = how_many * 2   # search wider next time
        tries = tries + 1

    return "I could not find a complete answer in our coffee pages. Which drink should I focus on?"


def level6_demo(model, embedder, index):
    """Run the full retry-until-it-passes loop on one comparison question."""

    print(careful_answer(
        model, embedder, index,
        "Compare ashwagandha coffee with turmeric coffee",
        ["ashwagandha", "turmeric"],
    ))


# ---------------------------------------------------------------------------
# BONUS — A FIFTH BRANCH: "SAFETY" (the notebook's homework exercise)
# ---------------------------------------------------------------------------
# If a question touches on pregnancy, medicine interactions, or general
# safety, answer ONLY from the "Benefits, Cautions & Practical Tips" section
# of each page (never from Method/Ingredients, etc.), and always add a
# closing disclaimer that this is not medical advice.
# ---------------------------------------------------------------------------

SAFETY_KEYWORDS = ["pregnant", "medicine", "safe"]
SAFETY_SECTION = "Benefits, Cautions & Practical Tips"


def is_safety_question(question):
    """Check if the question touches on a safety-sensitive topic."""

    q_lower = question.lower()
    for keyword in SAFETY_KEYWORDS:
        if keyword in q_lower:
            return True
    return False


def search_safety_only(embedder, index, question, how_many=4):
    """Like search(), but keeps only hits from the SAFETY_SECTION of each page."""

    question_vector = embedder.encode(question).tolist()
    result = index.query(
        vector=question_vector,
        top_k=how_many * 3,  # over-fetch, since we'll filter most of it back out
        namespace=NAMESPACE,
        include_metadata=True,
    )

    hits = []
    for match in result["matches"]:
        if match["metadata"]["section"] == SAFETY_SECTION:
            hit = {}
            hit["id"] = match["id"]
            hit["score"] = round(match["score"], 3)
            hit["title"] = match["metadata"]["title"]
            hit["section"] = match["metadata"]["section"]
            hit["text"] = match["metadata"]["text"]
            hits.append(hit)
        if len(hits) >= how_many:
            break
    return hits


def route_question(model, embedder, index, question):
    """L4/L5-style branch, now with a 5th 'safety' branch added on top."""

    if is_safety_question(question):
        hits = search_safety_only(embedder, index, question)
        prompt = build_prompt(question, hits)
        prompt = prompt + "\nAdd this line at the end: 'This is not medical advice.'"
        return ask_model(model, prompt)

    # Falls through to the plain Level-1-style search/answer for everything else.
    # (In a fuller build you would plug in the Level 4/5 branch + adaptive
    # logic here too - kept simple in the original notebook's homework answer.)
    hits = search(embedder, index, question)
    prompt = build_prompt(question, hits)
    return ask_model(model, prompt)


def safety_branch_demo(model, embedder, index):
    """Show the safety branch catching sensitive questions and disclaiming them."""

    questions = [
        "Tell me about ashwagandha coffee.",
        "Is ashwagandha coffee safe during pregnancy?",
        "What medicine interactions should I know about?",
        "Compare it with turmeric coffee.",
    ]
    for q in questions:
        print("\n", q)
        print(route_question(model, embedder, index, q))


# ---------------------------------------------------------------------------
# MAIN — run every level's demo in order
# ---------------------------------------------------------------------------
# Set UPLOAD_TO_PINECONE = True the FIRST time you run this script (so your
# coffee pages actually get embedded and stored). Set it back to False on
# every later run - the data stays in Pinecone between runs.
# ---------------------------------------------------------------------------

UPLOAD_TO_PINECONE = False


def main():
    ctx = setup()
    model, embedder, index = ctx["model"], ctx["embedder"], ctx["index"]

    # --- Cell 2 & 3: read the coffee pages, optionally upload them ---
    all_pieces = read_all_pages()

    if UPLOAD_TO_PINECONE:
        upload_pieces_to_pinecone(embedder, index, all_pieces)
    else:
        print("Skipping upload (UPLOAD_TO_PINECONE = False)")
    print(index.describe_index_stats())

    # --- Level 0: no RAG, watch it hallucinate ---
    print("\n=== LEVEL 0: no RAG ===")
    level0_demo(model)

    # --- Level 1: basic RAG ---
    print("\n=== LEVEL 1: basic RAG ===")
    level1_demo(model, embedder, index)

    # --- Level 2: prompt template ---
    print("\n=== LEVEL 2: prompt template ===")
    level2_demo(model, embedder, index)

    # --- Level 3: memory ---
    print("\n=== LEVEL 3: memory ===")
    level3_search_comparison_demo(embedder, index)
    level3_demo(model, embedder, index)

    # --- Level 4: branched RAG ---
    print("\n=== LEVEL 4: branched RAG ===")
    level4_routing_demo()
    level4_demo(model, embedder, index)

    # --- Level 5: adaptive RAG ---
    print("\n=== LEVEL 5: adaptive RAG ===")
    level5_demo(embedder, index)

    # --- Level 6: corrective RAG ---
    print("\n=== LEVEL 6: corrective RAG ===")
    level6_bad_answers_demo(embedder, index)
    level6_demo(model, embedder, index)

    # --- Bonus: safety branch ---
    print("\n=== BONUS: safety branch ===")
    safety_branch_demo(model, embedder, index)


if __name__ == "__main__":
    main()
