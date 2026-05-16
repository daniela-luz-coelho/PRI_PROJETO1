"""
boolean_search.py — pesquisa booleana sobre o índice invertido.

Suporta AND, OR, NOT com precedência correta (NOT > AND > OR),
AND implícito entre termos adjacentes, e parênteses para agrupar.

Exemplos:
    "information retrieval"         -> AND implícito
    "neural OR deep AND learning"   -> neural OR (deep AND learning)
    "(neural OR deep) AND learning" -> agrupamento explícito
    "machine NOT learning"          -> documentos com 'machine' sem 'learning'
"""

import logging
import re
from dataclasses import dataclass

from src.search.inverted_index import InvertedIndex, Posting

logger = logging.getLogger(__name__)

_AND = "AND"
_OR  = "OR"
_NOT = "NOT"
_LPAREN = "("
_RPAREN = ")"
_TERM   = "TERM"
_EOF    = "EOF"


@dataclass
class _Token:
    type: str
    value: str = ""


@dataclass
class SearchResult:
    doc_id: int
    document: dict
    score: float = 0.0


class BooleanSearchEngine:

    def __init__(self, index: InvertedIndex):
        self.index = index

    def search(self, query: str) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []

        logger.info("Boolean search: '%s'", query)
        try:
            tokens = self._tokenise_query(query)
            doc_ids = self._parse(tokens)
            results = self._build_results(doc_ids, query)
            logger.info("Found %d results for '%s'", len(results), query)
            return results
        except Exception as exc:
            logger.error("Error processing query '%s': %s", query, exc)
            return []

    def search_author(self, author_name: str) -> list[SearchResult]:
        docs = self.index.search_by_author(author_name)
        return [
            SearchResult(doc_id=self._find_doc_id(doc), document=doc, score=1.0)
            for doc in docs
        ]

    def _tokenise_query(self, query: str) -> list[_Token]:
        # espaços à volta dos parênteses para facilitar o split
        query = re.sub(r"\(", " ( ", query)
        query = re.sub(r"\)", " ) ", query)

        raw_tokens = query.split()
        tokens: list[_Token] = []

        for raw in raw_tokens:
            upper = raw.upper()
            if upper == "AND":
                tokens.append(_Token(_AND))
            elif upper == "OR":
                tokens.append(_Token(_OR))
            elif upper == "NOT":
                tokens.append(_Token(_NOT))
            elif raw == "(":
                tokens.append(_Token(_LPAREN))
            elif raw == ")":
                tokens.append(_Token(_RPAREN))
            else:
                tokens.append(_Token(_TERM, raw.strip('"'\'')))

        tokens = self._insert_implicit_and(tokens)
        tokens.append(_Token(_EOF))
        return tokens

    @staticmethod
    def _insert_implicit_and(tokens: list[_Token]) -> list[_Token]:
        # insere AND entre termos/grupos adjacentes sem operador explícito
        result: list[_Token] = []
        for i, tok in enumerate(tokens):
            result.append(tok)
            if i + 1 < len(tokens):
                curr_type = tok.type
                next_type = tokens[i + 1].type
                if curr_type in (_TERM, _RPAREN) and next_type in (_TERM, _LPAREN, _NOT):
                    result.append(_Token(_AND))
        return result

    # parser de descida recursiva: expr -> and_expr (OR and_expr)*
    #                              and_expr -> not_expr (AND not_expr)*
    #                              not_expr -> NOT not_expr | atom

    def _parse(self, tokens: list[_Token]) -> set[int]:
        self._tokens = tokens
        self._pos = 0
        return self._parse_or()

    def _current(self) -> _Token:
        return self._tokens[self._pos]

    def _consume(self, expected_type: str) -> _Token:
        tok = self._current()
        if tok.type != expected_type:
            raise SyntaxError(f"Expected {expected_type} but got {tok.type} ('{tok.value}')")
        self._pos += 1
        return tok

    def _parse_or(self) -> set[int]:
        left = self._parse_and()
        while self._current().type == _OR:
            self._pos += 1
            right = self._parse_and()
            left = left | right
        return left

    def _parse_and(self) -> set[int]:
        operands = [self._parse_not()]
        while self._current().type == _AND:
            self._pos += 1
            operands.append(self._parse_not())

        operands.sort(key=len)  # interseção mais eficiente se começar pela menor lista
        result = operands[0]
        for op in operands[1:]:
            result = result & op
            if not result:
                break  # se já vazio não vale a pena continuar
        return result

    def _parse_not(self) -> set[int]:
        if self._current().type == _NOT:
            self._pos += 1
            operand = self._parse_not()
            all_ids = set(self.index._documents.keys())
            return all_ids - operand
        return self._parse_atom()

    def _parse_atom(self) -> set[int]:
        tok = self._current()
        if tok.type == _TERM:
            self._pos += 1
            postings = self.index.get_postings(tok.value)
            return {p.doc_id for p in postings}
        if tok.type == _LPAREN:
            self._pos += 1
            result = self._parse_or()
            self._consume(_RPAREN)
            return result
        if tok.type == _EOF:
            return set()
        raise SyntaxError(f"Unexpected token: {tok.type} ('{tok.value}')")

    def _build_results(self, doc_ids: set[int], query: str) -> list[SearchResult]:
        # score simples: nº de termos da query que aparecem no documento
        stopwords_ops = {"and", "or", "not"}
        query_terms = [
            t for t in query.lower().split()
            if t not in stopwords_ops and t not in ("(", ")")
        ]
        processed_terms = []
        for t in query_terms:
            processed_terms.extend(self.index.preprocessor.process(t))

        results = []
        for doc_id in doc_ids:
            doc = self.index.get_document(doc_id)
            if doc is None:
                continue
            score = sum(
                1.0 for term in set(processed_terms)
                if any(p.doc_id == doc_id for p in self.index.get_postings_raw(term))
            )
            results.append(SearchResult(doc_id=doc_id, document=doc, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _find_doc_id(self, doc: dict) -> int:
        for doc_id, stored in self.index._documents.items():
            if stored is doc:
                return doc_id
        return -1
