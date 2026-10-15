"""
================================================================================
 DilSeVector Concierge — RAG Pipeline rebuilt on LangChain (local / VS Code)
================================================================================

This is the local-machine twin of `RAG_LangChain_Colab.ipynb`. Same 14 parts,
same LangChain components, same comments — just adapted to run from a
terminal instead of a Colab cell, with API keys read from a local `.env`
file instead of Colab's Secrets panel or key-text-files.

WHAT THIS SCRIPT WALKS THROUGH
--------------------------------
 Part 1   The one idea: everything is a Runnable
 Part 2   The pipe `|` and LCEL
 Part 3   Models and messages
 Part 4   Prompts: system vs human, few-shot
 Part 5   Output parsers, including JSON
 Part 6   Documents, loaders, splitters
 Part 7   Embeddings and vector stores (hybrid search)
 Part 8   Retrievers and the alpha knob
 Part 9   Smarter retrieval: multi-query, self-query, compression, diversity, budget
 Part 10  Rebuilding the Concierge as one LangChain chain
 Part 11  Conversation memory
 Part 12  Self-Adaptive RAG (plan -> answer -> verify -> retry)
 Part 13  Seeing inside your chain (debugging)

WHY LANGCHAIN (in one sentence)
--------------------------------
Session 5 hand-wired five different libraries (sentence-transformers,
pinecone, google-generativeai, plain Python) into one script. LangChain
gives every piece — models, prompts, parsers, retrievers — the same shape
(a "Runnable"), so they snap together with a pipe (`|`) instead of glue code,
and swapping a provider means changing one string instead of rewriting a
function.

IMPORTANT — LangChain has NO answer-checking tools
----------------------------------------------------
LangChain can search, prompt, call models, route, rerank, compress, and
remember. It has **nothing** that checks whether an answer is actually true.
The four checks from Session 5 (Evidence / Fair-citations / Coverage / Rules)
still have to be hand-written — see Part 10's `has_valid_citations()` and
your own Session 5 notebook for the full four-check version.

HOW TO RUN
----------
  1. Follow README.md to install requirements and create your .env file.
  2. Put the `coffee_pages/` folder (source HTML pages) next to this file.
  3. The FIRST time only, pass --upload so your data gets embedded and
     indexed into your Pinecone "coffee-hybrid" index. Skip it afterwards.
  4. Run:  python rag_langchain_pipeline.py
     Or run just a few parts:  python rag_langchain_pipeline.py --parts 1,2,10

Every function mirrors the notebook's own cells closely, so you can read this
script side by side with the Colab notebook and recognise every line.
================================================================================
"""

import os
import re
import glob
import inspect
import argparse

from dotenv import load_dotenv                          # reads secrets from a local .env file
from bs4 import BeautifulSoup                            # parses the coffee-page HTML files


# ==============================================================================
# SECTION 0 — ENVIRONMENT / API KEYS
# ==============================================================================
# Loads GOOGLE_API_KEY and PINECONE_API_KEY from a local .env file (see
# README.md). This replaces Colab's Secrets panel / key-text-files.
# ==============================================================================

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    raise RuntimeError(
        "GOOGLE_API_KEY not found. Create a .env file (see README.md) "
        "with GOOGLE_API_KEY=your-key-here."
    )
if not os.environ.get("PINECONE_API_KEY"):
    raise RuntimeError(
        "PINECONE_API_KEY not found. Create a .env file (see README.md) "
        "with PINECONE_API_KEY=your-key-here."
    )

# Where the source HTML pages live, relative to this script.
COFFEE_PAGES_DIR = "coffee_pages"

# Name of the Pinecone index used for hybrid (dense + sparse) search.
# Must already exist, created with the "dotproduct" metric, dimension
# matching the text-embedding-004 embedding model.
PINECONE_INDEX_NAME = "coffee-hybrid"

# Module-level objects that later parts reuse, same as the notebook's global
# variables (llm, embeddings, retriever, chunks, ...). They start as None and
# get filled in as each part's setup function runs, in order, from main().
llm = None
embeddings = None
pc = None
index = None
bm25 = None
retriever = None
plain_store = None
chunks = []          # list of langchain_core.documents.Document
shortener = None      # LLMChainExtractor used by the compression demos
concierge_prompt = None
concierge_chain = None
full_chain = None
history = []          # list of HumanMessage/AIMessage, used by Part 11's chat()


# ==============================================================================
# PART 1 — THE ONE IDEA: EVERYTHING IS A RUNNABLE
# ==============================================================================
# If you remember one thing from today, remember this: almost everything in
# LangChain is a "Runnable" — an object with the same three methods:
#   .invoke(x)   -> run it once, on one input
#   .batch([x1, x2, ...]) -> run it on a list of inputs
#   .stream(x)   -> run it, yielding pieces as they're ready
# Wrap any ordinary Python function in RunnableLambda and it gets all three
# for free. This is the shape that lets very different tools snap together.
# ==============================================================================

def part1_runnables_demo():
    from langchain_core.runnables import RunnableLambda

    print("\n" + "=" * 70)
    print("PART 1 — Everything is a Runnable")
    print("=" * 70)

    # an ordinary Python function
    def shout(text):
        return text.upper()

    # wrap it, and it becomes a Runnable
    shout_step = RunnableLambda(shout)

    print("invoke:", shout_step.invoke("turmeric coffee"))
    print("one at a time :", shout_step.invoke("turmeric coffee"))
    print("a whole list  :", shout_step.batch(["turmeric coffee", "filter coffee", "cold brew"]))

    print("piece by piece:", end=" ")
    for piece in shout_step.stream("turmeric coffee"):
        print(piece, end="")
    print()

    return shout_step


# ==============================================================================
# PART 2 — THE PIPE `|` AND LCEL
# ==============================================================================
# If two things both have .invoke(), and the output type of the first matches
# the input type of the second, you can join them with `|`. The joined thing
# is ALSO a Runnable — so it too has .invoke(), .batch(), and .stream(). This
# is what lets you build long pipelines (prompt | llm | parser) that all
# support the same three operations "for free".
# ==============================================================================

def part2_lcel_demo(shout_step):
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    print("\n" + "=" * 70)
    print("PART 2 — The pipe | and LCEL")
    print("=" * 70)

    def add_excitement(text):
        return text + "!!!"

    excite_step = RunnableLambda(add_excitement)

    pipeline = shout_step | excite_step

    print("shout only :", shout_step.invoke("turmeric coffee"))
    print("both steps :", pipeline.invoke("turmeric coffee"))

    print("the joined pipeline batches too:")
    print(pipeline.batch(["turmeric coffee", "cold brew"]))

    def make_question(text):
        return "Have you tried " + text.lower() + "?"

    longer = pipeline | RunnableLambda(make_question)
    print("\nthree steps:", longer.invoke("turmeric coffee"))

    # RunnablePassthrough - the "leave it alone" step. Sometimes a chain needs
    # a slot that just forwards its input unchanged (used heavily in Part 10).
    keep = RunnablePassthrough()
    print("passthrough gives back exactly what it got:", keep.invoke("turmeric coffee"))


# ==============================================================================
# PART 3 — MODELS AND MESSAGES
# ==============================================================================
# init_chat_model("provider:model") is LangChain's ONE constructor for ANY
# chat model. Swap the string for a different provider and every downstream
# chain keeps working unchanged. Conversations are lists of typed messages
# (SystemMessage / HumanMessage / AIMessage) instead of one glued-together
# string.
# ==============================================================================

def part3_setup_model():
    """Build the shared `llm` object every later part reuses."""
    global llm
    from langchain.chat_models import init_chat_model

    # One constructor, any provider. Swap "google_genai:gemini-2.0-flash" for
    # "anthropic:claude-..." or "openai:gpt-..." and nothing else changes.
    llm = init_chat_model("google_genai:gemini-2.0-flash", temperature=0)
    return llm


def part3_demo():
    from langchain_core.messages import SystemMessage, HumanMessage

    print("\n" + "=" * 70)
    print("PART 3 — Models and messages")
    print("=" * 70)

    conversation = [
        SystemMessage(content="You are a coffee expert. Answer in one short sentence."),
        HumanMessage(content="What is masala coffee?")
    ]

    reply = llm.invoke(conversation)

    print("type of reply:", type(reply).__name__)
    print("the text     :", reply.content)


# ==============================================================================
# PART 4 — PROMPTS
# ==============================================================================
# In Session 5 our prompt was one long f-string. A ChatPromptTemplate is the
# same idea, but structured: it knows about roles (system / human / ai), fills
# in {slots}, and — because it's a Runnable too — it pipes straight into the
# model. Few-shot examples (showing the model worked examples) are just
# another prompt piece you can slot into the template.
# ==============================================================================

def part4_prompts_demo():
    from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

    print("\n" + "=" * 70)
    print("PART 4 — Prompts")
    print("=" * 70)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a coffee expert. Answer in one short sentence."),
        ("human", "What is {drink}?")
    ])

    filled = prompt.invoke({"drink": "masala coffee"})

    print("prompt.invoke gives back messages:")
    for message in filled.messages:
        print("  ", type(message).__name__, "->", message.content)

    # A prompt is a Runnable too, so it pipes. Smallest real chain in LangChain:
    small_chain = prompt | llm
    result = small_chain.invoke({"drink": "filter coffee"})
    print("\ntype:", type(result).__name__)
    print("text:", result.content)

    # Few-shot: showing the model examples of the format you want, instead of
    # just describing it in words.
    example_shape = ChatPromptTemplate.from_messages([
        ("human", "{question}"),
        ("ai", "{answer}")
    ])

    examples = [
        {"question": "Is turmeric coffee caffeinated?",
         "answer": "No, it is caffeine free when made with milk alone [03_turmeric]."},
        {"question": "Is filter coffee strong?",
         "answer": "Yes, the chicory blend gives it a strong body [04_filter]."}
    ]

    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_shape,
        examples=examples
    )

    styled_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer in one sentence and always end with a [page] citation."),
        few_shot,
        ("human", "{question}")
    ])

    built = styled_prompt.invoke({"question": "Is masala coffee sweet?"})
    print("\nmessages sent to the model (few-shot):")
    for message in built.messages:
        print("  ", type(message).__name__, "->", message.content[:70])

    return prompt


# ==============================================================================
# PART 5 — OUTPUT PARSERS
# ==============================================================================
# The model returns a message OBJECT (reply.content is the text). A parser
# turns that object into something more directly usable: StrOutputParser
# gives you plain text; JsonOutputParser gives you a real Python dict you can
# index into (e.g. decision["branch"]) — useful for routing decisions.
# ==============================================================================

def part5_output_parsers_demo(prompt):
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    print("\n" + "=" * 70)
    print("PART 5 — Output parsers")
    print("=" * 70)

    parser = StrOutputParser()
    full_chain_demo = prompt | llm | parser
    answer = full_chain_demo.invoke({"drink": "cold brew"})
    print("type:", type(answer).__name__)
    print("text:", answer)

    # JSON out - useful for routing decisions you can act on in Python.
    router_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Decide what kind of coffee question this is.\n"
         'Reply with ONLY JSON like: {{"branch": "compare", "topics": ["turmeric", "masala"]}}\n'
         "branch must be one of: explain, compare, recommend, lookup."),
        ("human", "{question}")
    ])

    router_chain = router_prompt | llm | JsonOutputParser()

    print("\nJSON routing demo:")
    for question in ["Compare turmeric coffee with masala coffee",
                      "How much water for cold brew?"]:
        decision = router_chain.invoke({"question": question})
        print(decision, " <-", question)
        print("   branch is a real Python value:", decision["branch"])


# ==============================================================================
# PART 6 — DOCUMENTS, LOADERS AND SPLITTERS  (no API key needed)
# ==============================================================================
# A Document carries text (page_content) PLUS metadata (page id, section,
# etc.) as one object, instead of the parallel dictionaries we used in
# Session 5. Splitters cut long text into pieces small enough to embed
# sensibly, copying the parent's metadata onto every piece. For OUR coffee
# pages specifically, we split by <h2> SECTION instead of by raw character
# count — a judgement call, same as Session 5.
# ==============================================================================

def part6_documents_and_splitters_demo():
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print("\n" + "=" * 70)
    print("PART 6 — Documents, loaders and splitters")
    print("=" * 70)

    example = Document(
        page_content="Turmeric coffee is usually caffeine free when made with milk alone.",
        metadata={"page": "03_turmeric_coffee_haldi_cappuccino", "section": "Overview"}
    )
    print("text    :", example.page_content)
    print("metadata:", example.metadata)
    print("page id :", example.metadata["page"])

    # Generic splitter demo: cut a long blob into chunk_size-ish pieces, with
    # some overlap so context isn't lost at the boundary.
    long_text = ("Masala coffee is brewed with spices. " * 20).strip()
    big_document = Document(page_content=long_text, metadata={"page": "demo_page"})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    pieces = splitter.split_documents([big_document])
    print("\n1 document became", len(pieces), "pieces")
    for piece in pieces[:3]:
        print("  ", len(piece.page_content), "chars |", piece.metadata)


def read_one_page(filename):
    """
    Turn one coffee-page HTML file into a list of section Documents.
    Same logic as Session 5's read_one_page(), but returns LangChain
    Document objects (page_content + metadata) instead of plain dicts.
    """
    from langchain_core.documents import Document

    html = open(filename, encoding="utf-8", errors="ignore").read()
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "head"]):
        tag.decompose()

    title = soup.find("h1").get_text(strip=True)
    page_id = os.path.basename(filename).replace(".html", "")

    documents = []
    for heading in soup.find_all("h2"):
        section_name = heading.get_text(strip=True)

        section_text = ""
        for item in heading.next_siblings:
            if getattr(item, "name", None) == "h2":
                break
            if hasattr(item, "get_text"):
                section_text = section_text + " " + item.get_text(" ", strip=True)
        section_text = " ".join(section_text.split())

        if len(section_text) > 40:
            info = {"page": page_id, "title": title, "section": section_name}
            full_text = title + " - " + section_name + ". " + section_text
            documents.append(Document(page_content=full_text, metadata=info))

    return documents


def load_coffee_pages():
    """Read every HTML file under coffee_pages/ into a flat list of Documents."""
    global chunks
    chunks = []
    for filename in sorted(glob.glob(os.path.join(COFFEE_PAGES_DIR, "*.html"))):
        for document in read_one_page(filename):
            chunks.append(document)

    print("Documents made:", len(chunks))
    if chunks:
        print("Example:", chunks[0].metadata)
    return chunks


# ==============================================================================
# PART 7 — EMBEDDINGS AND VECTOR STORES
# ==============================================================================
# Embeddings turn text into a list of numbers (a vector) so "closeness in
# meaning" becomes "closeness in space". Here we use Google's
# text-embedding-004 model, and store the vectors in Pinecone using HYBRID
# search: a dense (meaning-based) vector plus a sparse (exact-keyword, BM25)
# vector, combined by an "alpha" knob (Part 8).
# ==============================================================================

def part7_setup_embeddings():
    """Build the shared `embeddings` object every later part reuses."""
    global embeddings
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return embeddings


def part7_embeddings_demo():
    print("\n" + "=" * 70)
    print("PART 7 — Embeddings and vector stores")
    print("=" * 70)

    vector = embeddings.embed_query("is turmeric coffee caffeine free")
    print("a question became", len(vector), "numbers")
    print("first five:", vector[:5])


def part7_setup_retriever(upload=False):
    """
    Build the shared hybrid retriever (dense + sparse search) backed by the
    Pinecone index PINECONE_INDEX_NAME, and optionally upload `chunks` into
    it (only needed the FIRST time — the data persists in Pinecone after
    that).
    """
    global pc, index, bm25, retriever

    from pinecone import Pinecone
    from pinecone_text.sparse import BM25Encoder
    from langchain_community.retrievers import PineconeHybridSearchRetriever

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(PINECONE_INDEX_NAME)

    texts = []
    for chunk in chunks:
        texts.append(chunk.page_content)

    # BM25 is the "sparse" half of hybrid search: exact keyword matching,
    # scored by how rare/important each word is across the corpus.
    bm25 = BM25Encoder().default()
    bm25.fit(texts)

    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25,
        index=index,
        top_k=5,
        alpha=0.5
    )

    if upload:
        metadatas = []
        for chunk in chunks:
            metadatas.append(chunk.metadata)
        retriever.add_texts(texts=texts, metadatas=metadatas)
        print("Uploaded", len(texts), "chunks into Pinecone index:", PINECONE_INDEX_NAME)
    else:
        print("Skipping upload (pass --upload the first time you run this script)")

    found = retriever.invoke("is turmeric coffee caffeine free")
    print("Returned", len(found), "Documents:")
    for document in found[:3]:
        print("  ", document.metadata["page"][:36], "|", document.metadata["section"])

    return retriever


# ==============================================================================
# PART 8 — RETRIEVERS AND THE ALPHA KNOB
# ==============================================================================
# A retriever is a Runnable that takes a question (a string) and returns
# Documents. Hybrid search blends two scores with a knob called ALPHA:
#   alpha = 0.0 -> exact words only (sparse / BM25)
#   alpha = 1.0 -> meaning only (dense / embeddings)
#   alpha = 0.5 -> a blend of both
# Never trust a comment about which end is which without checking — this demo
# proves it directly using pinecone_text's own scaling function.
# ==============================================================================

def part8_alpha_knob_demo():
    from pinecone_text.hybrid import hybrid_convex_scale

    print("\n" + "=" * 70)
    print("PART 8 — Retrievers and the alpha knob")
    print("=" * 70)

    print(inspect.getdoc(hybrid_convex_scale))

    dense_part, sparse_part = hybrid_convex_scale([1.0, 1.0], {"indices": [0], "values": [1.0]}, alpha=1.0)
    print("\nalpha = 1.0 -> dense", dense_part, "sparse", sparse_part["values"], " => MEANING ONLY")

    dense_part, sparse_part = hybrid_convex_scale([1.0, 1.0], {"indices": [0], "values": [1.0]}, alpha=0.0)
    print("alpha = 0.0 -> dense", dense_part, "sparse", sparse_part["values"], " => EXACT WORDS ONLY")

    # Now watch results actually move as alpha changes.
    question = "ashwagandha for stress"

    for alpha_value in [0.0, 0.5, 1.0]:
        retriever.alpha = alpha_value

        if alpha_value == 0.0:
            label = "exact words only"
        elif alpha_value == 1.0:
            label = "meaning only"
        else:
            label = "both"

        print("\nalpha =", alpha_value, "(" + label + ")")
        for document in retriever.invoke(question)[:3]:
            print("   ", document.metadata["page"][:36], "|", document.metadata["section"])

    retriever.alpha = 0.5   # always put the knob back afterwards


# ==============================================================================
# PART 9 — SMARTER RETRIEVAL
# ==============================================================================
# Ready-made retrievers and post-processors, each fixing a different failure
# mode of plain top-k search:
#   9.1  Multi-query / decomposition - search several wordings, or split a
#        multi-part question into sub-questions, and merge the results
#   9.2  Self-query - let the model write a metadata FILTER from plain English
#   9.3  Compression / reranking - shrink each Document down to just the
#        relevant sentences (Session 4's two-step retrieve-then-rerank idea)
#   9.4  Deduplication / diversity - drop near-identical results, or use MMR
#        to explicitly trade off relevance vs variety
#   9.5  Cheap transforms - fix typos/units or translate BEFORE you even
#        call an LLM-based retriever (not everything needs a model)
#   9.6  Context budget - cap how many Documents you actually send, by an
#        estimated token count
# ==============================================================================

def part9_1_multi_query_demo():
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever
    from langchain_core.prompts import PromptTemplate

    print("\n" + "=" * 70)
    print("PART 9.1 — Multi-query retrieval and decomposition")
    print("=" * 70)

    multi_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

    results = multi_retriever.invoke("what is that golden coloured drink and does it keep you awake")

    pages_found = []
    for document in results:
        page = document.metadata["page"]
        if page not in pages_found:
            pages_found.append(page)

    print("Pages found by searching several wordings:")
    for page in pages_found:
        print("  ", page)

    # Same class, a different job: decompose one complex question into 2-4
    # sub-questions, search each, and merge the results.
    decomp_prompt = PromptTemplate.from_template(
        "Decompose into 2-4 concise sub-questions that together answer: {question}\n"
        "Return one per line."
    )

    decomposer = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        prompt=decomp_prompt,
        include_original=True
    )

    results = decomposer.invoke(
        "Compare turmeric coffee and saffron coffee for taste and health effects"
    )

    pages_found = []
    for document in results:
        page = document.metadata["page"]
        if page not in pages_found:
            pages_found.append(page)

    print("\npages found by decomposing the question:")
    for page in pages_found:
        print("   ", page)


def part9_2_self_query_demo():
    """
    Self-query - the model reads the question and writes its own metadata
    filter (e.g. section = "Method") plus a semantic search string. Falls
    back to plain retrieval if the model's filter doesn't parse or match.
    """
    global plain_store
    from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
    from langchain_classic.chains.query_constructor.schema import AttributeInfo
    from langchain_pinecone import PineconeVectorStore

    print("\n" + "=" * 70)
    print("PART 9.2 — Self-query retrieval")
    print("=" * 70)

    plain_store = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

    field_info = [
        AttributeInfo(name="section", type="string",
                      description="One of: Overview, Ingredients (Base & Optional), Method, "
                                  "Variations & Regional Twists, Flavor Profile & Pairings, "
                                  "Benefits, Cautions & Practical Tips, Cultural & Historical Notes, FAQ"),
        AttributeInfo(name="page", type="string", description="Page id, e.g. 10_saffron_coffee_kesar_coffee"),
        AttributeInfo(name="title", type="string", description="Page title")
    ]

    self_query = SelfQueryRetriever.from_llm(
        llm, plain_store, "Indian coffee pages, split into sections", field_info, verbose=True)

    try:
        results = self_query.invoke("how do I make the saffron one")
        for document in results[:3]:
            print("  ", document.metadata.get("page", "?")[:34], "|", document.metadata.get("section"))
    except Exception as problem:
        print("Self-query failed:", type(problem).__name__)
        print("Falling back to normal search:")
        for document in retriever.invoke("how do I make saffron coffee")[:3]:
            print("  ", document.metadata["page"][:34], "|", document.metadata["section"])

    return plain_store


def part9_3_compression_demo():
    """
    Compression / reranking - LLMChainExtractor shrinks each Document down to
    only the sentences relevant to the question, cutting how much text (and
    cost) reaches the final prompt while keeping page ids in metadata.
    """
    global shortener
    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import LLMChainExtractor

    print("\n" + "=" * 70)
    print("PART 9.3 — Compression and reranking")
    print("=" * 70)

    shortener = LLMChainExtractor.from_llm(llm)

    retriever.top_k = 10
    tidy_retriever = ContextualCompressionRetriever(
        base_compressor=shortener,
        base_retriever=retriever
    )

    question = "is turmeric coffee caffeine free"

    before = retriever.invoke(question)
    after = tidy_retriever.invoke(question)

    before_size = 0
    for document in before:
        before_size = before_size + len(document.page_content)

    after_size = 0
    for document in after:
        after_size = after_size + len(document.page_content)

    print("before:", len(before), "pieces |", before_size, "characters")
    print("after :", len(after), "pieces |", after_size, "characters")

    print("\npage ids survived in metadata:")
    for document in after:
        print("  ", document.metadata.get("page", "MISSING"))

    retriever.top_k = 5
    return shortener


def part9_4_dedup_and_diversity_demo():
    """
    Deduplication and diversity:
      - EmbeddingsRedundantFilter drops near-identical pieces
      - EmbeddingsFilter keeps only the top-k most relevant
      - LongContextReorder puts the strongest evidence at the START and END
        of the context (models pay less attention to the middle)
      - MMR (maximal marginal relevance) trades relevance for variety directly
        at search time, instead of via a separate compressor step
    """
    from langchain_classic.retrievers.document_compressors import (
        DocumentCompressorPipeline,
        EmbeddingsFilter
    )
    from langchain_community.document_transformers import (
        EmbeddingsRedundantFilter,
        LongContextReorder
    )
    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

    print("\n" + "=" * 70)
    print("PART 9.4 — Deduplication and diversity")
    print("=" * 70)

    tidy_pipeline = DocumentCompressorPipeline(transformers=[
        EmbeddingsRedundantFilter(embeddings=embeddings, similarity_threshold=0.92),
        EmbeddingsFilter(embeddings=embeddings, k=6),
        LongContextReorder()
    ])

    diverse_retriever = ContextualCompressionRetriever(
        base_compressor=tidy_pipeline,
        base_retriever=retriever
    )

    retriever.top_k = 15
    question = "ashwagandha coffee benefits"

    before = retriever.invoke(question)
    after = diverse_retriever.invoke(question)

    print("before:", len(before), "pieces")
    print("after :", len(after), "pieces")
    for document in after:
        print("   ", document.metadata["page"][:36], "|", document.metadata["section"])

    retriever.top_k = 5

    # MMR - diversity at search time instead, directly on the plain vector store.
    if plain_store is not None:
        mmr_retriever = plain_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,              # how many we end up with
                "fetch_k": 30,       # how many we choose from
                "lambda_mult": 0.5   # 0 = maximum diversity, 1 = maximum relevance
            }
        )
        print("\nMMR results:")
        for document in mmr_retriever.invoke("ashwagandha coffee benefits"):
            print("   ", document.metadata.get("page", "?")[:36], "|", document.metadata.get("section"))
    else:
        print("\n(Skipping MMR demo - run Part 9.2 first to build `plain_store`.)")


def normalize_query(question):
    """Cheap, model-free query cleanups: fix common typos/unit spellings."""
    question = question.replace("celcius", "celsius")
    question = question.replace("miligrams", "milligrams")
    question = re.sub(r"\bdeg\b", "degrees", question)
    return question


def part9_5_cheap_transforms_demo():
    """
    Cheap transforms - not everything needs a model. Simple string fixes
    (normalize_query) cost nothing; translation genuinely needs an LLM call,
    but it's still just `prompt | llm | parser` - no special "translation"
    component required.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    print("\n" + "=" * 70)
    print("PART 9.5 — Cheap transforms")
    print("=" * 70)

    messy = "brew at 90 deg celcius for ashwagandha coffe"
    print("before:", messy)
    print("after :", normalize_query(messy))

    # Translation is a plain chain - note there is no LLMChain in v1.
    translate_prompt = ChatPromptTemplate.from_messages([
        ("human", "Translate to English, keep the meaning. Reply with only the translation: {text}")
    ])
    translate = translate_prompt | llm | StrOutputParser()

    hindi = "क्या केसर कॉफी मीठी होती है?"
    english = translate.invoke({"text": hindi})
    print("\ntranslated:", english)
    print("finds:", retriever.invoke(english)[0].metadata["page"])


def estimate_tokens(text):
    """Very rough token estimate: ~4 characters per token."""
    return max(1, len(text) // 4)


def fit_budget(documents, max_tokens=1200):
    """Keep adding Documents until the estimated token budget would be exceeded."""
    kept = []
    total = 0
    for document in documents:
        needed = estimate_tokens(document.page_content)
        if total + needed > max_tokens:
            break
        kept.append(document)
        total = total + needed
    return kept, total


def part9_6_budget_demo():
    print("\n" + "=" * 70)
    print("PART 9.6 — Fitting the context budget")
    print("=" * 70)

    retriever.top_k = 15
    candidates = retriever.invoke("ashwagandha coffee benefits")
    final_documents, used = fit_budget(candidates, max_tokens=600)

    print("retrieved:", len(candidates), "pieces")
    print("kept     :", len(final_documents), "pieces |", used, "estimated tokens")
    retriever.top_k = 5


# ==============================================================================
# PART 10 — REBUILDING THE CONCIERGE
# ==============================================================================
# Everything from Parts 1-9 comes together here: search -> format -> prompt ->
# generate -> extract text, expressed first as five explicit steps
# (answer_question), then compressed into one LCEL chain
# (prompt | llm | parser) that gets streaming and batching for free.
# ==============================================================================

def format_context(documents):
    """Turn retrieved Documents into text we can paste into a prompt, with [page] tags."""
    lines = []
    for document in documents:
        lines.append("[" + document.metadata["page"] + "] " + document.page_content)
    return "\n\n".join(lines)


def part10_setup_concierge():
    """Build the shared concierge_prompt / concierge_chain / full_chain objects."""
    global concierge_prompt, concierge_chain, full_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    concierge_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are the DilSeVector Concierge, a friendly coffee assistant.\n"
         "Answer ONLY from the CONTEXT below.\n"
         "Put the [page] at the end of every sentence that states a fact.\n"
         "If the CONTEXT does not answer the question, say you do not know. Never guess.\n"
         "Keep it under 80 words.\n\n"
         "CONTEXT:\n{context}"),
        ("human", "{question}")
    ])

    concierge_chain = concierge_prompt | llm | StrOutputParser()

    def search_and_format(question):
        documents = retriever.invoke(question)
        return format_context(documents)

    # The compact, idiomatic LangChain form: search included in the chain
    # itself. The input is now just the question string, not a dictionary.
    full_chain = (
        {"context": RunnableLambda(search_and_format), "question": RunnablePassthrough()}
        | concierge_prompt
        | llm
        | StrOutputParser()
    )

    return concierge_prompt, concierge_chain, full_chain


def answer_question(question):
    """The five RAG steps, written out explicitly (mirrors Session 5 exactly)."""
    documents = retriever.invoke(question)                     # 1 search
    context = format_context(documents)                        # 2 make it text
    filled = concierge_prompt.invoke({"context": context,
                                       "question": question})   # 3 fill the prompt
    reply = llm.invoke(filled)                                 # 4 ask the model
    return reply.content                                       # 5 take the text out


def answer_question_short(question):
    """Same five steps, using the pre-built concierge_chain (prompt | llm | parser)."""
    documents = retriever.invoke(question)
    context = format_context(documents)
    return concierge_chain.invoke({"context": context, "question": question})


def has_valid_citations(answer, documents):
    """Check that every [page] cited in the answer matches a page we actually retrieved."""
    pages = []
    for document in documents:
        pages.append(document.metadata["page"])

    found = re.findall(r"\[([^\]]+)\]", answer or "")
    if not found:
        return False
    for page in found:
        if page not in pages:
            return False
    return True


def part10_demo():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    print("\n" + "=" * 70)
    print("PART 10 — Rebuilding the Concierge")
    print("=" * 70)

    print(answer_question("Is turmeric coffee caffeine free?"))

    print("\n(compact form via concierge_chain):")
    print(answer_question_short("How do I make masala coffee?"))

    print("\n(fully chained form via full_chain, question-string input):")
    print(full_chain.invoke("What spices go into masala coffee?"))

    print("\nSTREAMING:\n")
    for piece in full_chain.stream("How do I make masala coffee?"):
        print(piece, end="")

    print("\n\nTWO QUESTIONS AT ONCE (batch):")
    for reply in full_chain.batch(["Is masala coffee sweet?", "What is jaggery coffee?"]):
        print("  -", reply[:90])

    # Enforcing citations: asking nicely in the prompt is not a guarantee.
    # Check, and repair with a second call if the check fails.
    repair_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite the answer so every factual sentence ends with a [page] citation.\n"
         "Use ONLY these page ids: {allowed}. Do not invent sources. Do not change the facts."),
        ("human", "{answer}")
    ])
    repair_chain = repair_prompt | llm | StrOutputParser()

    question = "How do I make masala coffee?"
    documents = retriever.invoke(question)
    answer = concierge_chain.invoke({
        "context": format_context(documents),
        "question": question
    })

    print("\ncited properly?", has_valid_citations(answer, documents))

    if not has_valid_citations(answer, documents):
        allowed = ", ".join(d.metadata["page"] for d in documents)
        answer = repair_chain.invoke({"allowed": allowed, "answer": answer})
        print("\nrepaired:\n", answer)


# ==============================================================================
# PART 11 — CONVERSATION MEMORY
# ==============================================================================
# Session 5's lesson still holds: HISTORY IS NOT MEMORY OF FACTS, it's just
# the conversation so far, glued onto the search so words like "it" still
# find the right page. Two ways to do it here:
#   11.1  Manual - a MessagesPlaceholder in the prompt, plus our own list
#   11.2  Managed - RunnableWithMessageHistory, which stores per-session
#         history for you (keyed by a session_id you choose)
# ==============================================================================

def part11_1_manual_memory_demo():
    """Build chat_chain + the module-level `history` list, and run two turns."""
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import HumanMessage, AIMessage

    global history

    print("\n" + "=" * 70)
    print("PART 11.1 — Manual conversation memory")
    print("=" * 70)

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are the DilSeVector Concierge. Answer ONLY from the CONTEXT.\n"
         "Put the [page] after every fact. Keep it under 80 words.\n\n"
         "CONTEXT:\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ])

    chat_chain = chat_prompt | llm | StrOutputParser()

    history = []

    def chat(question):
        # Glue the last question on, so words like "it" still find the right page.
        search_text = question
        if len(history) >= 2:
            search_text = history[-2].content + " " + question

        documents = retriever.invoke(search_text)
        context = format_context(documents)

        reply = chat_chain.invoke({"context": context, "history": history, "question": question})

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=reply))
        return reply

    print("Tell me about ashwagandha coffee.")
    print(chat("Tell me about ashwagandha coffee."))

    print("\nCompare it with turmeric coffee.")
    print(chat("Compare it with turmeric coffee."))

    print("\nMessages stored:", len(history))
    return chat_chain, chat


def part11_2_managed_memory_demo(chat_chain):
    """RunnableWithMessageHistory - LangChain manages per-session history for you."""
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.chat_history import InMemoryChatMessageHistory

    print("\n" + "=" * 70)
    print("PART 11.2 — The managed form: RunnableWithMessageHistory")
    print("=" * 70)

    sessions = {}

    def get_session_history(session_id):
        if session_id not in sessions:
            sessions[session_id] = InMemoryChatMessageHistory()
        return sessions[session_id]

    chat_with_history = RunnableWithMessageHistory(
        chat_chain,
        get_session_history,
        input_messages_key="question",       # matches ("human", "{question}")
        history_messages_key="history"       # matches MessagesPlaceholder("history")
    )

    config = {"configurable": {"session_id": "user_001"}}

    documents = retriever.invoke("What is beaten coffee?")
    print(chat_with_history.invoke(
        {"context": format_context(documents), "question": "What is beaten coffee?"},
        config=config
    ))

    print("\nmessages stored:", len(get_session_history("user_001").messages))
    # To start fresh without touching the documents:
    #   sessions["user_001"] = InMemoryChatMessageHistory()


# ==============================================================================
# PART 12 — SELF-ADAPTIVE RAG
# ==============================================================================
# Everything so far ran the SAME pipeline for every question. Here, plan_query
# looks at the question first and picks settings (top_k, alpha, whether to
# compress); answer_once runs with those settings; and adaptive_answer adds a
# second LLM call that VERIFIES the answer against the context, retrying
# wider automatically if confidence is low or the model said "I don't know".
# ==============================================================================

def plan_query(question):
    """Crude keyword-based planner: pick top_k / alpha / compress from the question shape."""
    lowered = question.lower()

    broad_words = ["compare", "vs", "guide", "how", "steps", "difference"]
    is_broad = False
    for word in broad_words:
        if word in lowered:
            is_broad = True

    is_exact = False
    for character in lowered:
        if character.isdigit():
            is_exact = True

    if is_broad:
        return {"top_k": 12, "alpha": 0.8, "compress": True}
    if is_exact:
        return {"top_k": 8, "alpha": 0.2, "compress": False}
    return {"top_k": 8, "alpha": 0.5, "compress": True}


def part12_plan_query_demo():
    print("\n" + "=" * 70)
    print("PART 12 — Self-Adaptive RAG: plan_query()")
    print("=" * 70)

    for question in ["Compare turmeric vs saffron coffee",
                      "What temperature for a 90 second brew?",
                      "What is jaggery coffee"]:
        print(plan_query(question), " <-", question)


def answer_once(question, plan):
    """Run one retrieval + answer pass using a specific plan's settings."""
    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

    retriever.top_k = plan["top_k"]
    retriever.alpha = plan["alpha"]

    if plan["compress"] and shortener is not None:
        source = ContextualCompressionRetriever(
            base_compressor=shortener,
            base_retriever=retriever
        )
    else:
        source = retriever

    documents = source.invoke(question)
    context = format_context(documents)
    answer = concierge_chain.invoke({"context": context, "question": question})
    return answer, documents


def part12_adaptive_answer_demo():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    print("\n" + "=" * 70)
    print("PART 12 — Self-Adaptive RAG: verify + retry")
    print("=" * 70)

    verify_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict verifier. Compare the ANSWER to the CONTEXT.\n"
         'Reply with ONLY JSON: {{"supported": true, "confidence": 0.0, "problems": []}}\n'
         "confidence is 0.0 to 1.0, judged from the CONTEXT alone."),
        ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}")
    ])

    verify_chain = verify_prompt | llm | JsonOutputParser()

    def adaptive_answer(question):
        plan = plan_query(question)
        answer, documents = answer_once(question, plan)

        # A model asked for JSON does not always return JSON - never let the
        # verifier take down the answer we already have.
        try:
            verdict = verify_chain.invoke({
                "context": format_context(documents),
                "question": question,
                "answer": answer
            })
        except Exception as problem:
            print("   (verifier did not return JSON:", type(problem).__name__, "- keeping the answer)")
            verdict = {"supported": True, "confidence": 1.0, "problems": []}

        confidence = verdict.get("confidence", 0.0)
        unknown = "do not know" in answer.lower()

        if confidence < 0.5 or unknown:
            print("   (low confidence -> retrying wider)")
            wider = {"top_k": plan["top_k"] + 8, "alpha": 0.5, "compress": False}
            answer, documents = answer_once(question, wider)
            plan = wider

        retriever.top_k = 5      # always put the knobs back
        retriever.alpha = 0.5
        return answer, plan, verdict

    final_answer, used_plan, verdict = adaptive_answer(
        "Compare turmeric coffee and saffron coffee"
    )
    print("PLAN USED:", used_plan)
    print("CONFIDENCE:", verdict.get("confidence"))
    print("\nANSWER:\n", final_answer)

    return adaptive_answer


# ==============================================================================
# PART 13 — SEEING INSIDE YOUR CHAIN
# ==============================================================================
# The honest cost of a framework: when something looks wrong, you can no
# longer just read top-to-bottom Python. Test any single step on its own
# (retriever.invoke), or peek at the exact messages a prompt built
# (prompt.invoke(...).messages) before blaming the model.
# For a full execution trace across every step, LangSmith is the standard
# tool (needs its own API key - left commented out below).
# ==============================================================================

def part13_debugging_demo():
    print("\n" + "=" * 70)
    print("PART 13 — Seeing inside your chain")
    print("=" * 70)

    # 1. test any single step on its own
    documents = retriever.invoke("is masala coffee sweet")
    print("step 1 gave", len(documents), "documents")

    # 2. look at the exact prompt the model received
    context = format_context(documents)
    filled = concierge_prompt.invoke({"context": context, "question": "Is masala coffee sweet?"})
    print("\n--- what the model actually sees ---")
    print(filled.messages[0].content[:400], "...")
    print("\n--- the human turn ---")
    print(filled.messages[1].content)

    # 3. LangSmith (uncomment and add a key to try it)
    # os.environ["LANGCHAIN_TRACING_V2"] = "true"
    # os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
    # full_chain.invoke("Is masala coffee sweet?")   # then open smith.langchain.com


# ==============================================================================
# MAIN — runs the whole story end to end, part by part
# ==============================================================================

ALL_PARTS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]


def main():
    parser = argparse.ArgumentParser(description="DilSeVector Concierge - LangChain RAG pipeline")
    parser.add_argument("--upload", action="store_true",
                         help="Embed coffee_pages/ and upload into the Pinecone hybrid index "
                              "before running the demos (only needed the FIRST time).")
    parser.add_argument("--parts", type=str, default=",".join(ALL_PARTS),
                         help="Comma-separated list of parts to run, e.g. '1,2,10'. "
                              "Defaults to all parts (1-13).")
    args = parser.parse_args()

    wanted = set(p.strip() for p in args.parts.split(","))

    # Parts 3, 7, 6/10 build shared objects (llm, embeddings, chunks, retriever,
    # concierge_prompt/chain) that later parts depend on, so we always set
    # those up regardless of which parts were asked for.
    part3_setup_model()
    part7_setup_embeddings()
    load_coffee_pages()
    part7_setup_retriever(upload=args.upload)

    shout_step = None

    if "1" in wanted:
        shout_step = part1_runnables_demo()
    if "2" in wanted:
        if shout_step is None:
            from langchain_core.runnables import RunnableLambda
            shout_step = RunnableLambda(lambda text: text.upper())
        part2_lcel_demo(shout_step)
    if "3" in wanted:
        part3_demo()

    prompt = None
    if "4" in wanted:
        prompt = part4_prompts_demo()
    if "5" in wanted:
        if prompt is None:
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a coffee expert. Answer in one short sentence."),
                ("human", "What is {drink}?")
            ])
        part5_output_parsers_demo(prompt)
    if "6" in wanted:
        part6_documents_and_splitters_demo()
    if "7" in wanted:
        part7_embeddings_demo()
    if "8" in wanted:
        part8_alpha_knob_demo()
    if "9" in wanted:
        part9_1_multi_query_demo()
        part9_2_self_query_demo()
        part9_3_compression_demo()
        part9_4_dedup_and_diversity_demo()
        part9_5_cheap_transforms_demo()
        part9_6_budget_demo()

    # Part 10 must run before 11/12/13 (they reuse concierge_prompt/chain).
    if "10" in wanted or "11" in wanted or "12" in wanted or "13" in wanted:
        part10_setup_concierge()
    if "10" in wanted:
        part10_demo()

    chat_chain = None
    if "11" in wanted:
        chat_chain, _ = part11_1_manual_memory_demo()
        part11_2_managed_memory_demo(chat_chain)
    if "12" in wanted:
        # Part 12's compression step benefits from Part 9.3's `shortener`.
        if shortener is None:
            part9_3_compression_demo()
        part12_plan_query_demo()
        part12_adaptive_answer_demo()
    if "13" in wanted:
        part13_debugging_demo()


if __name__ == "__main__":
    main()
