from fastapi import FastAPI, Query, HTTPException
from typing import List, Optional
import json

from src.config import settings
from src.search.tfidf import TFIDFEngine
from src.search.preprocessor import make_stemming_preprocessor

app = FastAPI(title="IR Search Engine")

# -------------------------------------------------------------------
# Load documents
# -------------------------------------------------------------------

def load_documents():
    try:
        with open(settings.DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


DOCUMENTS = load_documents()

# -------------------------------------------------------------------
# Engine
# -------------------------------------------------------------------

engine = TFIDFEngine(
    preprocessor=make_stemming_preprocessor(settings.DEFAULT_LANGUAGE),
    tf_scheme=settings.TF_SCHEME,
    use_sklearn=False,
)

engine.build_from_documents(DOCUMENTS)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def format_result(r):
    return {
        "doc_id": r.doc_id,
        "title": r.document.get("title", ""),
        "abstract": r.document.get("abstract", ""),
        "authors": r.document.get("authors", []),
        "year": r.document.get("year", ""),
        "doi": r.document.get("doi", ""),
        "document_link": r.document.get("document_link", ""),
        "score": r.score,
    }

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok"}


# ---------------- SEARCH ----------------

@app.get("/search")
def search(
    q: str = Query(...),
    top_k: int = settings.TOP_K_DEFAULT,
    algorithm: str = "custom",
):
    engine.use_sklearn = (algorithm == "sklearn")
    engine.build_from_documents(DOCUMENTS)

    results = engine.search(q, top_k=top_k)

    return {
        "query": q,
        "total": len(results),
        "results": [format_result(r) for r in results],
    }


# ---------------- BOOLEAN SEARCH (simplified fallback) ----------------

@app.get("/search/boolean")
def boolean_search(q: str = Query(...)):
    results = engine.search(q, top_k=settings.TOP_K_DEFAULT)

    return {
        "query": q,
        "total": len(results),
        "results": [format_result(r) for r in results],
    }


# ---------------- AUTHOR SEARCH ----------------

@app.get("/search/author")
def search_author(name: str = Query(...)):
    name_lower = name.lower()

    results = []
    for doc_id, doc in enumerate(DOCUMENTS):
        authors = " ".join(doc.get("authors", [])).lower()
        if name_lower in authors:
            results.append({
                "doc_id": doc_id,
                "title": doc.get("title", ""),
                "score": 1.0
            })

    return {
        "query": name,
        "total": len(results),
        "results": results,
    }


# ---------------- DOCUMENT BY ID ----------------

@app.get("/documents/{doc_id}")
def get_document(doc_id: int):
    if doc_id < 0 or doc_id >= len(DOCUMENTS):
        raise HTTPException(status_code=404, detail="Document not found")

    doc = DOCUMENTS[doc_id]
    doc["doc_id"] = doc_id
    return doc


# ---------------- STATS ----------------

@app.get("/stats")
def stats():
    return {
        "num_documents": len(DOCUMENTS),
        "vocabulary_size": len(engine._vocabulary),
    }


# ---------------- TF-IDF IDF ----------------

@app.get("/tfidf/idf")
def idf(term: str):
    return {
        "term": term,
        "idf": engine.get_term_idf(term)
    }