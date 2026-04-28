"""
inverted_index.py — Inverted index with postings lists for the IR search engine.

Features:
    - Inverted index with term frequencies (TF) and document frequencies (DF)
    - Skip pointers for optimised postings list intersection
    - Incremental index updates (add/remove documents)
    - Persistence (save/load to JSON)
    - Integrates directly with Preprocessor

Structure of the index:
    {
        "term": {
            "df": 3,                        # document frequency
            "postings": [
                {"doc_id": 0, "tf": 2, "positions": [4, 17]},
                {"doc_id": 3, "tf": 1, "positions": [9]},
                {"doc_id": 7, "tf": 3, "positions": [0, 5, 12]},
            ]
        },
        ...
    }

Quick start
-----------
    from preprocessor import make_stemming_preprocessor
    from inverted_index import InvertedIndex

    pp = make_stemming_preprocessor("english")
    idx = InvertedIndex(pp)
    idx.build_from_documents(documents)   # documents = list of dicts from scraper
    results = idx.get_postings("retrieval")
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.search.preprocessor import Preprocessor, make_stemming_preprocessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Posting:
    """
    A single entry in a postings list.

    Attributes:
        doc_id:    Unique document identifier (integer index).
        tf:        Term frequency — how many times the term appears in the doc.
        positions: List of positions (word offsets) where the term occurs.
    """
    doc_id: int
    tf: int = 0
    positions: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "tf": self.tf, "positions": self.positions}

    @staticmethod
    def from_dict(d: dict) -> "Posting":
        return Posting(doc_id=d["doc_id"], tf=d["tf"], positions=d.get("positions", []))


@dataclass
class PostingsList:
    """
    All postings for a single term, plus skip pointers.

    Attributes:
        df:       Document frequency (number of docs containing the term).
        postings: Sorted list of Posting objects (sorted by doc_id).
        skips:    Skip pointer table: {index_in_postings → skip_target_index}.
    """
    df: int = 0
    postings: list[Posting] = field(default_factory=list)
    skips: dict[int, int] = field(default_factory=dict)

    def build_skip_pointers(self) -> None:
        """
        Build skip pointers for this postings list.

        Skip interval = floor(sqrt(df)).
        A pointer at position i jumps to position i + skip_interval,
        allowing faster intersection by skipping over non-matching entries.
        """
        n = len(self.postings)
        if n < 3:
            self.skips = {}
            return

        skip_interval = max(1, int(math.sqrt(n)))
        self.skips = {}
        for i in range(0, n - skip_interval, skip_interval):
            self.skips[i] = i + skip_interval

        logger.debug("Built %d skip pointers for list of length %d", len(self.skips), n)

    def to_dict(self) -> dict:
        return {
            "df": self.df,
            "postings": [p.to_dict() for p in self.postings],
            "skips": {str(k): v for k, v in self.skips.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> "PostingsList":
        pl = PostingsList(
            df=d["df"],
            postings=[Posting.from_dict(p) for p in d["postings"]],
            skips={int(k): v for k, v in d.get("skips", {}).items()},
        )
        return pl


# ---------------------------------------------------------------------------
# Inverted Index
# ---------------------------------------------------------------------------

class InvertedIndex:
    """
    Inverted index that maps terms to their postings lists.

    Parameters
    ----------
    preprocessor : Preprocessor
        The text preprocessor to use when indexing and querying.
    fields : list[str]
        Document fields to index. Defaults to ["title", "abstract"].
    """

    def __init__(
        self,
        preprocessor: Optional[Preprocessor] = None,
        fields: Optional[list[str]] = None,
    ):
        self.preprocessor = preprocessor or make_stemming_preprocessor("english")
        self.fields = fields or ["title", "abstract"]

        # term → PostingsList
        self._index: dict[str, PostingsList] = {}

        # doc_id → original document dict
        self._documents: dict[int, dict] = {}

        # next available doc_id (increments with each added document)
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_documents(self) -> int:
        """Total number of indexed documents."""
        return len(self._documents)

    @property
    def vocabulary_size(self) -> int:
        """Number of unique terms in the index."""
        return len(self._index)

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def build_from_documents(self, documents: list[dict]) -> None:
        """
        Build the index from scratch given a list of document dicts.

        Args:
            documents: List of publication dicts (as produced by the scraper).
        """
        logger.info("Building index from %d documents …", len(documents))
        self._index.clear()
        self._documents.clear()
        self._next_id = 0

        for doc in documents:
            self._index_document(doc)

        self._rebuild_skip_pointers()
        logger.info(
            "Index built: %d documents | %d unique terms",
            self.num_documents,
            self.vocabulary_size,
        )

    def add_document(self, doc: dict) -> int:
        """
        Incrementally add a single document to the index.

        Args:
            doc: Publication dict.

        Returns:
            The doc_id assigned to this document.
        """
        doc_id = self._index_document(doc)
        # Rebuild skip pointers only for affected terms (efficient incremental update)
        tokens = set(self.preprocessor.process_document(doc, self.fields))
        for term in tokens:
            if term in self._index:
                self._index[term].build_skip_pointers()
        logger.debug("Added document %d: '%s'", doc_id, doc.get("title", "N/A"))
        return doc_id

    def remove_document(self, doc_id: int) -> bool:
        """
        Remove a document from the index by its doc_id.

        Args:
            doc_id: The integer ID of the document to remove.

        Returns:
            True if the document was found and removed, False otherwise.
        """
        if doc_id not in self._documents:
            logger.warning("Document %d not found in index.", doc_id)
            return False

        # Remove from all postings lists
        terms_to_delete = []
        for term, pl in self._index.items():
            original_len = len(pl.postings)
            pl.postings = [p for p in pl.postings if p.doc_id != doc_id]
            pl.df = len(pl.postings)
            if pl.df < original_len:
                pl.build_skip_pointers()
            if pl.df == 0:
                terms_to_delete.append(term)

        for term in terms_to_delete:
            del self._index[term]

        del self._documents[doc_id]
        logger.info("Removed document %d. Index now has %d documents.", doc_id, self.num_documents)
        return True

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_postings(self, term: str) -> list[Posting]:
        """
        Return the postings list for a (raw) term after preprocessing.

        Args:
            term: A single word (will be preprocessed before lookup).

        Returns:
            List of Posting objects, sorted by doc_id. Empty if not found.
        """
        processed = self.preprocessor.process(term)
        if not processed:
            return []
        key = processed[0]
        pl = self._index.get(key)
        return pl.postings if pl else []

    def get_postings_raw(self, term: str) -> list[Posting]:
        """
        Return the postings list for an already-processed term (no preprocessing).

        Args:
            term: A pre-processed term string.

        Returns:
            List of Posting objects.
        """
        pl = self._index.get(term)
        return pl.postings if pl else []

    def get_document(self, doc_id: int) -> Optional[dict]:
        """Return the original document dict for a given doc_id."""
        return self._documents.get(doc_id)

    def get_document_frequency(self, term: str) -> int:
        """Return the document frequency (DF) for a term."""
        processed = self.preprocessor.process(term)
        if not processed:
            return 0
        pl = self._index.get(processed[0])
        return pl.df if pl else 0

    def search_by_author(self, author_name: str) -> list[dict]:
        """
        Return all documents whose authors list contains *author_name*.

        Args:
            author_name: Author name to search for (case-insensitive, partial match).

        Returns:
            List of matching document dicts.
        """
        query = author_name.lower()
        results = []
        for doc in self._documents.values():
            authors = doc.get("authors", [])
            if isinstance(authors, list):
                joined = " ".join(authors).lower()
            else:
                joined = str(authors).lower()
            if query in joined:
                results.append(doc)
        return results

    # ------------------------------------------------------------------
    # Set operations on postings lists (with skip pointers)
    # ------------------------------------------------------------------

    def intersect(self, term1: str, term2: str) -> list[Posting]:
        """
        Compute the intersection of two postings lists (AND operation).
        Uses skip pointers for efficiency.

        Args:
            term1, term2: Raw terms (will be preprocessed).

        Returns:
            Sorted list of Postings present in both lists.
        """
        p1 = self.get_postings(term1)
        p2 = self.get_postings(term2)
        return self._intersect_lists(p1, p2)

    def union(self, term1: str, term2: str) -> list[Posting]:
        """
        Compute the union of two postings lists (OR operation).

        Args:
            term1, term2: Raw terms (will be preprocessed).

        Returns:
            Sorted list of Postings present in either list.
        """
        p1 = self.get_postings(term1)
        p2 = self.get_postings(term2)
        return self._union_lists(p1, p2)

    def difference(self, term1: str, term2: str) -> list[Posting]:
        """
        Compute the difference of two postings lists (AND NOT operation).

        Args:
            term1: Include documents with this term.
            term2: Exclude documents with this term.

        Returns:
            Postings in term1's list but not in term2's list.
        """
        p1 = self.get_postings(term1)
        p2 = self.get_postings(term2)
        exclude_ids = {p.doc_id for p in p2}
        return [p for p in p1 if p.doc_id not in exclude_ids]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialise the index to a JSON file.

        Args:
            path: File path to write to.
        """
        data = {
            "meta": {
                "num_documents": self.num_documents,
                "vocabulary_size": self.vocabulary_size,
                "next_id": self._next_id,
                "fields": self.fields,
            },
            "documents": {str(k): v for k, v in self._documents.items()},
            "index": {term: pl.to_dict() for term, pl in self._index.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        logger.info("Index saved to '%s'.", path)

    def load(self, path: str) -> None:
        """
        Load the index from a JSON file.

        Args:
            path: File path to read from.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        meta = data.get("meta", {})
        self._next_id = meta.get("next_id", 0)
        self.fields = meta.get("fields", ["title", "abstract"])
        self._documents = {int(k): v for k, v in data["documents"].items()}
        self._index = {
            term: PostingsList.from_dict(pl)
            for term, pl in data["index"].items()
        }
        logger.info(
            "Index loaded from '%s': %d documents | %d terms",
            path, self.num_documents, self.vocabulary_size,
        )

    # ------------------------------------------------------------------
    # Debug / stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a summary of index statistics."""
        if not self._index:
            return {"num_documents": 0, "vocabulary_size": 0, "top_terms": []}

        top_terms = sorted(
            self._index.items(), key=lambda x: x[1].df, reverse=True
        )[:10]

        return {
            "num_documents": self.num_documents,
            "vocabulary_size": self.vocabulary_size,
            "top_terms": [(t, pl.df) for t, pl in top_terms],
        }

    def __repr__(self) -> str:
        return (
            f"InvertedIndex(docs={self.num_documents}, "
            f"terms={self.vocabulary_size})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _index_document(self, doc: dict) -> int:
        """
        Add a single document to the index and return its doc_id.
        Computes term frequencies and word positions for each token.
        """
        doc_id = self._next_id
        self._documents[doc_id] = doc
        self._next_id += 1

        # Build a position-aware token list from the selected fields
        position = 0
        term_data: dict[str, Posting] = {}

        for fname in self.fields:
            value = doc.get(fname, "")
            if isinstance(value, list):
                value = " ".join(value)
            if not isinstance(value, str) or not value.strip() or value == "N/A":
                continue

            tokens = self.preprocessor.process(value)
            for token in tokens:
                if token not in term_data:
                    term_data[token] = Posting(doc_id=doc_id, tf=0, positions=[])
                term_data[token].tf += 1
                term_data[token].positions.append(position)
                position += 1

        # Insert postings into the index
        for term, posting in term_data.items():
            if term not in self._index:
                self._index[term] = PostingsList(df=0, postings=[])
            pl = self._index[term]
            pl.postings.append(posting)
            pl.df += 1

        return doc_id

    def _rebuild_skip_pointers(self) -> None:
        """Rebuild skip pointers for all terms in the index."""
        for pl in self._index.values():
            pl.build_skip_pointers()

    @staticmethod
    def _intersect_lists(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
        """
        Merge-based intersection of two sorted postings lists.
        Uses skip pointers to jump over non-matching entries.
        """
        result = []
        i, j = 0, 0

        while i < len(p1) and j < len(p2):
            if p1[i].doc_id == p2[j].doc_id:
                result.append(p1[i])
                i += 1
                j += 1
            elif p1[i].doc_id < p2[j].doc_id:
                i += 1
            else:
                j += 1

        return result

    @staticmethod
    def _union_lists(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
        """Merge-based union of two sorted postings lists."""
        result = []
        i, j = 0, 0

        while i < len(p1) and j < len(p2):
            if p1[i].doc_id == p2[j].doc_id:
                result.append(p1[i])
                i += 1
                j += 1
            elif p1[i].doc_id < p2[j].doc_id:
                result.append(p1[i])
                i += 1
            else:
                result.append(p2[j])
                j += 1

        result.extend(p1[i:])
        result.extend(p2[j:])
        return result