"""
rag_eval.py

Evaluate your RAG system (ingest.py + chatbot.py + direct_all.py) on ragsample.pdf.

- Builds chunks using the same logic as ingest.py (CharacterTextSplitter).
- If FAISS index doesn't exist, runs ingest.ingest_pdfs() to create it.
- Loads the FAISS index and the RAG chain (chatbot.setup_chatbot).
- For a set of queries it:
    * derives ground-truth relevant chunk IDs by keyword matching against chunks,
    * retrieves top-k documents via the retriever and maps them to chunk IDs,
    * optionally generates answers via the RAG chain,
    * computes retrieval metrics and faithfulness using direct_all.RAGEvaluator.
"""

import os
import sys
from typing import List, Dict, Tuple
from pathlib import Path

# re-use your provided evaluation class
from direct_all import RAGEvaluator

# reuse ingestion and chatbot code
import ingest
import chatbot

# langchain utilities for building chunks (same as in ingest.py)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# FAISS retriever load (used indirectly via chatbot.setup_chatbot)
# but we will also use vector_db.as_retriever() when needed
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------
# Helpers to build chunks
# -------------------------
def build_chunks_from_pdf(pdf_path: str, chunk_size: int = 1500, chunk_overlap: int = 500) -> Tuple[List[Dict], List[str]]:
    """
    Load the PDF and split into chunks using the same settings as ingest.py.
    Returns:
        chunks_meta: list of dicts with keys {id, page_content, metadata}
        chunk_texts: list of page_content strings in same order
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    chunks_meta = []
    chunk_texts = []
    for i, doc in enumerate(chunks):
        cid = f"chunk_{i}"
        page_content = doc.page_content
        metadata = dict(doc.metadata) if hasattr(doc, "metadata") else {}
        chunks_meta.append({"id": cid, "text": page_content, "metadata": metadata})
        chunk_texts.append(page_content)
    return chunks_meta, chunk_texts

# -------------------------
# Ground-truth derivation
# -------------------------
def derive_relevant_chunk_ids(chunks_meta: List[Dict], keywords: List[str], match_all: bool = True) -> List[str]:
    """
    Simple ground-truth: select chunk ids that contain the provided keywords.
    If match_all=True, a chunk must contain all keywords; else any keyword.
    """
    kws = [k.lower() for k in keywords]
    relevant = []
    for c in chunks_meta:
        text = (c["text"] or "").lower()
        if match_all:
            if all(k in text for k in kws):
                relevant.append(c["id"])
        else:
            if any(k in text for k in kws):
                relevant.append(c["id"])
    return relevant

# -------------------------
# Map retrieved documents to our chunk IDs
# -------------------------
def map_docs_to_chunk_ids(retrieved_docs, chunks_meta: List[Dict]) -> List[str]:
    """
    retrieved_docs: list of langchain Document objects (have .page_content)
    We map by exact substring match of page_content to chunk text. If exact
    not found, fallback to fuzzy containment.
    """
    chunk_text_to_id = {c["text"]: c["id"] for c in chunks_meta}
    mapped_ids = []
    for doc in retrieved_docs:
        txt = doc.page_content
        # exact
        cid = chunk_text_to_id.get(txt)
        if cid:
            mapped_ids.append(cid)
            continue
        # fallback: check containment
        found = None
        for c in chunks_meta:
            if txt.strip() and txt.strip() in c["text"]:
                found = c["id"]
                break
            if c["text"].strip() and c["text"].strip() in txt:
                found = c["id"]
                break
        if found:
            mapped_ids.append(found)
        else:
            # last resort: hash/snippet-based id
            mapped_ids.append(f"unknown:{hash(txt) & 0xffffffff:08x}")
    return mapped_ids

# -------------------------
# Main evaluation flow
# -------------------------
def main(
    pdf_path: str = "/mnt/data/ragsample.pdf",
    pdf_dir: str = "pdfs",
    index_path: str = "faiss_index",
    top_k: int = 5
):
    # ensure pdf_dir exists and contains the pdf (ingest expects pdfs/)
    pdf_dir_path = Path(pdf_dir)
    pdf_dir_path.mkdir(exist_ok=True)
    # copy if user provided a single path like /mnt/data/ragsample.pdf
    if not any(pdf_dir_path.glob("*.pdf")):
        # try to copy the provided pdf into pdfs/
        try:
            from shutil import copy2
            copy2(pdf_path, pdf_dir_path / Path(pdf_path).name)
            print(f"Copied {pdf_path} -> {pdf_dir}/{Path(pdf_path).name}")
        except Exception as e:
            print(f"Warning: could not copy PDF to {pdf_dir}: {e}")

    # Build canonical chunks from PDF (same logic ingest.py uses)
    print("Building chunks from PDF (for ground-truth mapping)...")
    # Try to find a PDF in pdf_dir (prefer explicit pdf_path if exists)
    chosen_pdf = pdf_path
    if not Path(chosen_pdf).exists():
        pdfs_in_dir = list(pdf_dir_path.glob("*.pdf"))
        if not pdfs_in_dir:
            print("No PDF found to build chunks. Please place ragsample.pdf in 'pdfs/' or set pdf_path.")
            sys.exit(1)
        chosen_pdf = str(pdfs_in_dir[0])
    chunks_meta, chunk_texts = build_chunks_from_pdf(chosen_pdf)
    print(f"Built {len(chunks_meta)} chunks.")

    # If index not present, run ingestion to create it
    if not Path(index_path).exists() or not any(Path(index_path).iterdir()):
        print("FAISS index not found, running ingest.ingest_pdfs() to create it...")
        ingest.ingest_pdfs(pdf_dir=pdf_dir, index_path=index_path)
    else:
        print("FAISS index found. Skipping ingestion.")

    # Load the RAG chain (chatbot)
    print("Loading RAG chain (chatbot.setup_chatbot)...")
    try:
        rag_chain = chatbot.setup_chatbot(index_path=index_path)
    except Exception as e:
        print("Warning: couldn't set up full chatbot (LLM may be missing).")
        print("Attempting to load vector DB retriever directly for retrieval-only evaluation.")
        rag_chain = None

    # Load the vector DB to access retriever directly if needed
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    # Instantiate evaluator
    evaluator = RAGEvaluator(use_llm_for_faithfulness=False)

    # Define evaluation queries and expected keyword-based ground truth.
    # These queries are chosen based on ragsample.pdf contents.
    eval_items = [
        {
            "query": "When was TechCorp founded?",
            "keywords": ["founded in 2010", "founding", "2010"],
            "match_all": False
        },
        {
            "query": "Where is TechCorp headquarters located?",
            "keywords": ["1234 innovation drive", "san francisco", "headquarters"],
            "match_all": False
        },
        {
            "query": "What is SecureVault?",
            "keywords": ["securevault", "cybersecurity", "end-to-end encryption"],
            "match_all": True
        },
        {
            "query": "What products does TechCorp offer?",
            "keywords": ["CloudAI Platform", "DataSync Pro", "SecureVault"],
            "match_all": False
        },
        {
            "query": "What initiative did TechCorp announce in March 2024?",
            "keywords": ["march 2024", "carbon neutral", "green technology initiative"],
            "match_all": False
        },
        {
            "query": "Who is the CEO of TechCorp?",
            "keywords": ["sarah johnson", "ceo", "co-founder"],
            "match_all": False
        },
        {
            "query": "What is the price of SecureVault for large enterprises?",
            "keywords": ["50,000", "50 000", "large enterprises", "annual subscription"],
            "match_all": False
        },
        {
            "query": "Which Asian office locations does TechCorp have?",
            "keywords": ["tokyo", "singapore", "bangalore", "shanghai"],
            "match_all": False
        }
    ]

    # Run evaluation across queries
    all_queries = []
    all_retrieved_ids = []
    all_relevant_ids = []
    all_generated_answers = []
    all_retrieved_contexts = []
    all_relevance_scores = []

    for item in eval_items:
        q = item["query"]
        print("\n" + "-"*60)
        print(f"Evaluating query: {q}")

        # derive ground-truth relevant chunks
        relevant_ids = derive_relevant_chunk_ids(chunks_meta, item["keywords"], match_all=item.get("match_all", True))
        print(f"  Derived {len(relevant_ids)} ground-truth relevant chunk(s): {relevant_ids}")

        # retrieve top_k docs using retriever
        retrieved_docs = retriever.get_relevant_documents(q)
        retrieved_ids = map_docs_to_chunk_ids(retrieved_docs, chunks_meta)
        retrieved_texts = [d.page_content for d in retrieved_docs]
        print(f"  Retrieved {len(retrieved_ids)} doc id(s): {retrieved_ids}")

        # generated answer via rag_chain if available
        generated_answer = None
        try:
            if rag_chain is not None:
                response = rag_chain.invoke({"input": q})
                # different chain implementations may return 'answer' or 'output' keys:
                generated_answer = response.get("answer") or response.get("output") or response.get("result") or str(response)
                print(f"  Generated answer: {str(generated_answer)[:200]}...")
            else:
                print("  Skipping generation (rag_chain not available).")
        except Exception as e:
            print("  Warning: generation failed:", e)
            generated_answer = None

        # Build simple relevance scores dict (1.0 for ground-truth)
        relevance_scores = {rid: 1.0 if rid in relevant_ids else 0.0 for rid in retrieved_ids}

        # Evaluate using RAGEvaluator
        metrics = evaluator.evaluate_query(
            query=q,
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            generated_answer=generated_answer,
            retrieved_contexts=retrieved_texts,
            relevance_scores=relevance_scores
        )

        # Collect for dataset-level averaging
        all_queries.append(q)
        all_retrieved_ids.append(retrieved_ids)
        all_relevant_ids.append(relevant_ids)
        all_generated_answers.append(generated_answer)
        all_retrieved_contexts.append(retrieved_texts)
        all_relevance_scores.append(relevance_scores)

    # Compute averaged metrics
    avg_metrics = evaluator.evaluate_dataset(
        queries=all_queries,
        retrieved_ids_list=all_retrieved_ids,
        relevant_ids_list=all_relevant_ids,
        generated_answers=all_generated_answers,
        retrieved_contexts_list=all_retrieved_contexts,
        relevance_scores_list=all_relevance_scores
    )

    # Print results
    evaluator.print_results()
    print("\n" + "="*60)
    print("AVERAGE METRICS ACROSS ALL QUERIES")
    for k, v in avg_metrics.items():
        try:
            print(f"{k}: {v:.4f}")
        except Exception:
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
