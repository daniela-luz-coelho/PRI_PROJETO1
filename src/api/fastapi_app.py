"""
fastapi_app.py — FastAPI REST API for the IR Search Engine.

Endpoints:
    GET  /search          — Free-text TF-IDF search
    GET  /search/boolean  — Boolean query search (AND/OR/NOT)
    GET  /search/author   — Search by author name
    GET  /documents/{id}  — Get document by ID
    GET  /stats           — Index statistics
    GET  /filters/values  — List available filter values (years, types, subjects)
    GET  /tfidf/idf       — IDF value for a term
    GET  /docs            — OpenAPI Swagger UI (auto-generated)

Run:
    uvicorn src.api.fastapi_app:app --reload --host 0.0.0.0 --port 8000
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.search.preprocessor import (
    make_lemmatisation_preprocessor,
    make_stemming_preprocessor,
)
from src.search.inverted_index import InvertedIndex
from src.search.boolean_search import BooleanSearchEngine
from src.search.tfidf import TFIDFEngine
from src.search.term_document_matrix import TermDocumentMatrix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data path
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent.parent / "scraper" / "scraper_results.json"


def _load_documents() -> list[dict]:
    if not DATA_PATH.exists():
        logger.warning("Data file not found: %s — using empty corpus", DATA_PATH)
        return []
    with open(DATA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Global search engines (initialised at startup)
# ---------------------------------------------------------------------------

_engines: dict = {}


def _build_engines(documents: list[dict]) -> dict:
    """Build all search engines from documents."""
    logger.info("Building search engines for %d documents …", len(documents))

    pp_stem = make_stemming_preprocessor("english")
    pp_lemma = make_lemmatisation_preprocessor("english")

    tfidf_custom = TFIDFEngine(pp_stem, use_sklearn=False, tf_scheme="log")
    tfidf_custom.build_from_documents(documents)

    tfidf_sklearn = TFIDFEngine(pp_lemma, use_sklearn=True, tf_scheme="log")
    tfidf_sklearn.build_from_documents(documents)

    inv_index = InvertedIndex(pp_stem)
    inv_index.build_from_documents(documents)
    boolean_engine = BooleanSearchEngine(inv_index)

    # Term-document matrix (binary, for boolean retrieval and educational display)
    tdm = TermDocumentMatrix(pp_stem, mode="binary")
    tdm.build_from_documents(documents)


    return {
        "tfidf_custom": tfidf_custom,
        "tfidf_sklearn": tfidf_sklearn,
        "boolean": boolean_engine,
        "inverted_index": inv_index,
        "tdm": tdm,
        "documents": documents,
    }


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    documents = _load_documents()
    _engines.update(_build_engines(documents))
    logger.info("Search engines ready.")
    yield


app = FastAPI(
    title="RepositóriUM Search Engine",
    description="Motor de pesquisa de publicações científicas da Universidade do Minho.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class DocumentOut(BaseModel):
    doc_id: int
    title: str
    authors: list[str]
    abstract: str
    year: str
    doi: str
    document_link: str
    doc_type: str
    subject: str
    score: float


class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[DocumentOut]


class StatsResponse(BaseModel):
    num_documents: int
    vocabulary_size: int
    index_terms: int


class FilterValuesResponse(BaseModel):
    years: list[str]
    doc_types: list[str]
    subjects: list[str]


# ---------------------------------------------------------------------------
# Filter helper
# ---------------------------------------------------------------------------

def _apply_filters(
    doc: dict,
    year_from: Optional[int],
    year_to: Optional[int],
    doc_type: Optional[str],
    subject: Optional[str],
) -> bool:
    """
    Return True if the document passes all active filters.

    Filters are only applied when a value is provided.
    Fields not present in the document are treated as unknown and pass through.
    """
    # --- year filter ---
    if year_from is not None or year_to is not None:
        try:
            y = int(doc.get("year", 0))
            if year_from is not None and y < year_from:
                return False
            if year_to is not None and y > year_to:
                return False
        except (ValueError, TypeError):
            pass  # unparseable year → don't exclude

    # --- document type filter ---
    if doc_type:
        doc_type_val = doc.get("doc_type", doc.get("type", ""))
        if doc_type_val and doc_type.lower() not in doc_type_val.lower():
            return False

    # --- subject / area filter ---
    if subject:
        subject_val = doc.get("subject", doc.get("area", doc.get("keywords", "")))
        if isinstance(subject_val, list):
            subject_val = " ".join(subject_val)
        if subject_val and subject.lower() not in subject_val.lower():
            return False

    return True


# ---------------------------------------------------------------------------
# Document serialiser
# ---------------------------------------------------------------------------

def _doc_to_out(doc: dict, doc_id: int, score: float) -> DocumentOut:
    authors = doc.get("authors", [])
    if isinstance(authors, str):
        authors = [authors]

    # subject may be a list (keywords) or a string
    subject_raw = doc.get("subject", doc.get("area", doc.get("keywords", "N/A")))
    if isinstance(subject_raw, list):
        subject_raw = "; ".join(subject_raw)

    return DocumentOut(
        doc_id=doc_id,
        title=doc.get("title", "N/A"),
        authors=authors,
        abstract=doc.get("abstract", "N/A"),
        year=doc.get("year", "N/A"),
        doi=doc.get("doi", "N/A"),
        document_link=doc.get("document_link", ""),
        doc_type=doc.get("doc_type", doc.get("type", "N/A")),
        subject=subject_raw,
        score=score,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["info"])
def root():
    """Health check."""
    return {"status": "ok", "message": "RepositóriUM Search Engine API"}


@app.get("/search", response_model=SearchResponse, tags=["search"])
def search_freetext(
    q: str = Query(..., description="Free-text query"),
    top_k: int = Query(20, ge=1, le=100, description="Max results"),
    algorithm: str = Query("custom", description="'custom' or 'sklearn'"),
    use_stemming: bool = Query(True, description="Use stemming (true) or lemmatisation (false)"),
    remove_stopwords: bool = Query(True, description="Remove stop words"),
    year_from: Optional[int] = Query(None, description="Filter: minimum publication year"),
    year_to: Optional[int] = Query(None, description="Filter: maximum publication year"),
    doc_type: Optional[str] = Query(None, description="Filter: document type (e.g. 'thesis', 'article')"),
    subject: Optional[str] = Query(None, description="Filter: subject area or keyword"),
):
    """
    Free-text search using TF-IDF + cosine similarity.

    - **algorithm**: `custom` (own implementation) or `sklearn`
    - **use_stemming**: Porter stemming vs. WordNet lemmatisation
    - **doc_type**: partial match on the document type field (if available)
    - **subject**: partial match on the subject/area/keywords field (if available)
    """
    engine_key = "tfidf_sklearn" if algorithm == "sklearn" else "tfidf_custom"
    engine: TFIDFEngine = _engines.get(engine_key)
    if engine is None:
        raise HTTPException(503, "Search engine not ready")

    raw_results = engine.search(q, top_k=top_k * 5)  # over-fetch to allow for filtering

    results = []
    for r in raw_results:
        if not _apply_filters(r.document, year_from, year_to, doc_type, subject):
            continue
        results.append(_doc_to_out(r.document, r.doc_id, r.score))
        if len(results) >= top_k:
            break

    return SearchResponse(query=q, total=len(results), results=results)


@app.get("/search/boolean", response_model=SearchResponse, tags=["search"])
def search_boolean(
    q: str = Query(..., description="Boolean query (AND, OR, NOT, parentheses)"),
    top_k: int = Query(20, ge=1, le=100),
    year_from: Optional[int] = Query(None, description="Filter: minimum publication year"),
    year_to: Optional[int] = Query(None, description="Filter: maximum publication year"),
    doc_type: Optional[str] = Query(None, description="Filter: document type (e.g. 'thesis', 'article')"),
    subject: Optional[str] = Query(None, description="Filter: subject area or keyword"),
):
    """
    Boolean search supporting AND, OR, NOT operators.

    Examples:
    - `machine learning`          → implicit AND
    - `neural OR deep`            → OR
    - `learning NOT deep`         → AND NOT
    - `(neural OR deep) AND network` → grouped
    """
    engine: BooleanSearchEngine = _engines.get("boolean")
    if engine is None:
        raise HTTPException(503, "Search engine not ready")

    raw_results = engine.search(q)

    results = []
    for r in raw_results:
        if not _apply_filters(r.document, year_from, year_to, doc_type, subject):
            continue
        results.append(_doc_to_out(r.document, r.doc_id, r.score))
        if len(results) >= top_k:
            break

    return SearchResponse(query=q, total=len(results), results=results)


@app.get("/search/author", response_model=SearchResponse, tags=["search"])
def search_author(
    name: str = Query(..., description="Author name (partial match)"),
    top_k: int = Query(20, ge=1, le=100),
    year_from: Optional[int] = Query(None, description="Filter: minimum publication year"),
    year_to: Optional[int] = Query(None, description="Filter: maximum publication year"),
    doc_type: Optional[str] = Query(None, description="Filter: document type (e.g. 'thesis', 'article')"),
    subject: Optional[str] = Query(None, description="Filter: subject area or keyword"),
):
    """
    Search for documents by author name (partial, case-insensitive).
    Supports the same date, type and subject filters as the other endpoints.
    """
    engine: BooleanSearchEngine = _engines.get("boolean")
    if engine is None:
        raise HTTPException(503, "Search engine not ready")

    raw_results = engine.search_author(name)

    results = []
    for r in raw_results:
        if not _apply_filters(r.document, year_from, year_to, doc_type, subject):
            continue
        results.append(_doc_to_out(r.document, r.doc_id, r.score))
        if len(results) >= top_k:
            break

    return SearchResponse(query=name, total=len(results), results=results)


@app.get("/documents/{doc_id}", response_model=DocumentOut, tags=["documents"])
def get_document(doc_id: int):
    """Retrieve a document by its internal ID."""
    engine: TFIDFEngine = _engines.get("tfidf_custom")
    if engine is None:
        raise HTTPException(503, "Search engine not ready")
    doc = engine.get_document(doc_id)
    if doc is None:
        raise HTTPException(404, f"Document {doc_id} not found")
    return _doc_to_out(doc, doc_id, 0.0)


@app.get("/stats", response_model=StatsResponse, tags=["info"])
def get_stats():
    """Return index statistics."""
    engine: TFIDFEngine = _engines.get("tfidf_custom")
    inv: InvertedIndex = _engines.get("inverted_index")
    if engine is None:
        raise HTTPException(503, "Search engine not ready")
    return StatsResponse(
        num_documents=engine.num_documents,
        vocabulary_size=len(engine._vocabulary),
        index_terms=inv.vocabulary_size if inv else 0,
    )


@app.get("/filters/values", response_model=FilterValuesResponse, tags=["info"])
def get_filter_values():
    """
    Return the set of distinct values available for each filter field.

    Useful for populating dropdowns in the frontend.
    Fields not present in documents are returned as empty lists.
    """
    documents: list[dict] = _engines.get("documents", [])

    years: set[str] = set()
    doc_types: set[str] = set()
    subjects: set[str] = set()

    for doc in documents:
        if y := doc.get("year"):
            years.add(str(y))

        if t := doc.get("doc_type", doc.get("type")):
            doc_types.add(str(t))

        raw_subj = doc.get("subject", doc.get("area", doc.get("keywords")))
        if raw_subj:
            if isinstance(raw_subj, list):
                subjects.update(raw_subj)
            else:
                subjects.add(str(raw_subj))

    return FilterValuesResponse(
        years=sorted(years, reverse=True),
        doc_types=sorted(doc_types),
        subjects=sorted(subjects),
    )


@app.get("/tfidf/idf", tags=["info"])
def get_idf(term: str = Query(..., description="Term to look up")):
    """Return IDF value for a term (custom engine)."""
    engine: TFIDFEngine = _engines.get("tfidf_custom")
    if engine is None:
        raise HTTPException(503, "Search engine not ready")
    idf_val = engine.get_term_idf(term)
    return {"term": term, "idf": idf_val}


# ---------------------------------------------------------------------------
# Term-Document Matrix endpoints
# ---------------------------------------------------------------------------

@app.get("/matrix/search", response_model=SearchResponse, tags=["matrix"])
def matrix_boolean_search(
    q: str = Query(..., description="Boolean query (AND, OR, NOT, parentheses)"),
    top_k: int = Query(20, ge=1, le=100),
    year_from: Optional[int] = Query(None, description="Filter: minimum publication year"),
    year_to: Optional[int] = Query(None, description="Filter: maximum publication year"),
    doc_type: Optional[str] = Query(None, description="Filter: document type"),
    subject: Optional[str] = Query(None, description="Filter: subject area"),
):
    """
    Boolean search using the term-document matrix (bitwise operations on row vectors).

    This is the classic Boolean retrieval model from Manning et al. Chapter 1.
    Results are equivalent to /search/boolean but computed via matrix operations
    instead of postings list traversal — useful for educational comparison.
    """
    tdm: TermDocumentMatrix = _engines.get("tdm")
    if tdm is None:
        raise HTTPException(503, "Term-document matrix not ready")

    raw_results = tdm.boolean_search(q)

    results = []
    for r in raw_results:
        if not _apply_filters(r.document, year_from, year_to, doc_type, subject):
            continue
        results.append(_doc_to_out(r.document, r.doc_id, r.score))
        if len(results) >= top_k:
            break

    return SearchResponse(query=q, total=len(results), results=results)


@app.get("/matrix/stats", tags=["matrix"])
def matrix_stats():
    """
    Return term-document matrix statistics.

    Includes dimensions, sparsity and the top terms by document frequency.
    """
    tdm: TermDocumentMatrix = _engines.get("tdm")
    if tdm is None:
        raise HTTPException(503, "Term-document matrix not ready")
    return tdm.stats()


@app.get("/matrix/term", tags=["matrix"])
def matrix_term_vector(
    term: str = Query(..., description="Term to look up (raw, will be preprocessed)"),
):
    """
    Return the binary presence vector for a term across all documents.

    The vector has one entry per document: 1 if the term appears, 0 otherwise.
    Useful for educational visualisation of how a term is distributed.
    """
    tdm: TermDocumentMatrix = _engines.get("tdm")
    if tdm is None:
        raise HTTPException(503, "Term-document matrix not ready")

    vector = tdm.get_term_vector(term)
    if vector is None:
        raise HTTPException(404, f"Term '{term}' not found in vocabulary after preprocessing")

    return {
        "term": term,
        "doc_ids": tdm.doc_ids,
        "vector": vector.tolist(),
        "document_frequency": int(vector.sum()),
    }


@app.get("/matrix/document", tags=["matrix"])
def matrix_document_vector(doc_id: int):
    """
    Return the term weight vector for a specific document.

    Each entry corresponds to a vocabulary term: its value is 1 (binary mode)
    if the term is present in the document, 0 otherwise.
    """
    tdm: TermDocumentMatrix = _engines.get("tdm")
    if tdm is None:
        raise HTTPException(503, "Term-document matrix not ready")

    vector = tdm.get_document_vector(doc_id)
    if vector is None:
        raise HTTPException(404, f"Document {doc_id} not found in matrix")

    non_zero_terms = [
        {"term": tdm.vocabulary[i], "weight": float(vector[i])}
        for i in range(len(vector)) if vector[i] > 0
    ]

    return {
        "doc_id": doc_id,
        "num_terms": len(non_zero_terms),
        "terms": non_zero_terms,
    }


@app.get("/matrix/preview", tags=["matrix"])
def matrix_preview(
    max_terms: int = Query(20, ge=1, le=100, description="Max terms to show (rows)"),
    max_docs: int = Query(10, ge=1, le=50, description="Max documents to show (columns)"),
):
    """
    Return a human-readable preview of the term-document matrix.

    Slices the matrix to the top N terms (by document frequency) and first M
    documents. Suitable for educational display in the frontend.
    """
    tdm: TermDocumentMatrix = _engines.get("tdm")
    if tdm is None:
        raise HTTPException(503, "Term-document matrix not ready")
    if tdm.matrix is None:
        raise HTTPException(503, "Matrix not built")

    import numpy as np

    # Select top terms by document frequency
    df_per_term = (tdm.matrix > 0).sum(axis=1)  # shape: (n_terms,)
    top_row_indices = np.argsort(df_per_term)[::-1][:max_terms]
    top_row_indices = sorted(top_row_indices.tolist())

    # Slice columns
    col_slice = min(max_docs, len(tdm.doc_ids))
    col_indices = list(range(col_slice))

    # Build preview
    terms_shown = [tdm.vocabulary[i] for i in top_row_indices]
    doc_ids_shown = [tdm.doc_ids[j] for j in col_indices]
    documents: list[dict] = _engines.get("documents", [])
    doc_titles = []
    for doc_id in doc_ids_shown:
        doc = next((d for i, d in enumerate(documents) if i == doc_id), {})
        doc_titles.append(doc.get("title", f"Doc {doc_id}")[:40])

    submatrix = tdm.matrix[np.ix_(top_row_indices, col_indices)]

    return {
        "terms": terms_shown,
        "doc_ids": doc_ids_shown,
        "doc_titles": doc_titles,
        "matrix": submatrix.tolist(),
        "mode": tdm.mode,
        "total_terms": len(tdm.vocabulary),
        "total_docs": len(tdm.doc_ids),
    }