"""
tfidf.py — Motor de pesquisa vetorial com TF-IDF e similaridade de cosseno.

A ideia central é representar documentos e queries como vetores num espaço
de alta dimensão (uma dimensão por termo), onde o peso de cada dimensão é
o seu score TF-IDF. A relevância de um documento para uma query é então
a similaridade de cosseno entre os dois vetores.

Mantemos dois backends que produzem resultados equivalentes:
  - custom: implementação própria com dicionários esparsos. Mais transparente.
  - sklearn: usa TfidfVectorizer + matrizes scipy.sparse. Mais eficiente.
"""

import logging
import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from src.search.preprocessor import Preprocessor, make_stemming_preprocessor

logger = logging.getLogger(__name__)

TFScheme = Literal["raw", "log", "boolean"]


@dataclass
class TFIDFResult:
    """Resultado individual de uma pesquisa TF-IDF."""
    doc_id: int
    document: dict
    score: float


@dataclass
class SimilarityResult:
    """Resultado de similaridade entre documentos."""
    doc_id: int
    document: dict
    similarity: float


class TFIDFEngine:
    """
    Motor de pesquisa vetorial com dois backends (custom e sklearn).

    Parameters
    ----------
    preprocessor : Preprocessor
        Pipeline NLP a aplicar a documentos e queries.
    fields : list[str]
        Campos a indexar (default: title + abstract).
    tf_scheme : "raw" | "log" | "boolean"
        Esquema de TF a usar. "log" é o recomendado para IR geral.
    use_sklearn : bool
        Se True, usa o backend sklearn. Se False, usa o custom.
    """

    def __init__(
        self,
        preprocessor: Optional[Preprocessor] = None,
        fields: Optional[list] = None,
        tf_scheme: TFScheme = "log",
        use_sklearn: bool = False,
    ):
        self.preprocessor = preprocessor or make_stemming_preprocessor("english")
        self.fields = fields or ["title", "abstract"]
        self.tf_scheme = tf_scheme
        self.use_sklearn = use_sklearn

        # Estado interno — separado por backend
        self._documents: dict = {}
        self._doc_vectors: dict = {}   # backend custom: {doc_id: {term: weight}}
        self._idf: dict = {}
        self._vocabulary: set = set()
        self._next_id: int = 0

        self._sklearn_vectorizer = None
        self._sklearn_matrix = None
        self._sklearn_doc_ids: list = []

    # ------------------------------------------------------------------
    # Construção do índice
    # ------------------------------------------------------------------

    def build_from_documents(self, documents: list) -> None:
        """
        (Re)constrói o índice TF-IDF a partir de uma lista de documentos.
        Limpa qualquer estado anterior.
        """
        self._documents.clear()
        self._doc_vectors.clear()
        self._idf.clear()
        self._vocabulary.clear()
        self._next_id = 0

        # Primeira passagem: tokenizar e construir o vocabulário
        token_lists: dict = {}
        for doc in documents:
            doc_id = self._next_id
            self._documents[doc_id] = doc
            self._next_id += 1
            tokens = self.preprocessor.process_document(doc, self.fields)
            token_lists[doc_id] = tokens
            self._vocabulary.update(tokens)

        N = len(self._documents)

        if self.use_sklearn:
            self._build_sklearn(documents)
        else:
            # Segunda passagem: calcular IDF e vectores TF-IDF
            self._idf = self._compute_idf(token_lists, N)
            for doc_id, tokens in token_lists.items():
                self._doc_vectors[doc_id] = self._compute_tfidf_vector(tokens)

        logger.info("Índice TF-IDF construído: %d docs | %d termos", N, len(self._vocabulary))

    # ------------------------------------------------------------------
    # Pesquisa
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> list:
        """
        Pesquisa por similaridade de cosseno.

        Args:
            query:  Texto da query em linguagem natural.
            top_k:  Número máximo de resultados a devolver.

        Returns:
            Lista de TFIDFResult ordenada por score descendente.
        """
        if not self._documents or not query.strip():
            return []
        if self.use_sklearn:
            return self._search_sklearn(query, top_k)
        return self._search_custom(query, top_k)

    def _search_custom(self, query: str, top_k: int) -> list:
        """Pesquisa usando o backend custom (vetores esparsos em dicts)."""
        query_tokens = self.preprocessor.process(query)
        if not query_tokens:
            return []
        query_vector = self._compute_tfidf_vector(query_tokens)
        if not query_vector:
            return []

        results = []
        for doc_id, doc_vector in self._doc_vectors.items():
            score = self._cosine_similarity(query_vector, doc_vector)
            if score > 0:
                results.append(
                    TFIDFResult(doc_id=doc_id, document=self._documents[doc_id], score=round(score, 6))
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Helpers do backend custom
    # ------------------------------------------------------------------

    def _compute_tfidf_vector(self, tokens: list) -> dict:
        """
        Calcula o vetor TF-IDF esparso para uma lista de tokens.
        Devolve apenas os termos com peso > 0 (vetor esparso).
        """
        if not tokens:
            return {}
        tf_raw: dict = {}
        for token in tokens:
            tf_raw[token] = tf_raw.get(token, 0) + 1
        vector: dict = {}
        for term, raw_tf in tf_raw.items():
            tf_weight = self._apply_tf_scheme(raw_tf)
            idf_weight = self._idf.get(term, 0.0)
            weight = tf_weight * idf_weight
            if weight > 0:
                vector[term] = weight
        return vector

    def _apply_tf_scheme(self, raw_tf: int) -> float:
        """Aplica o esquema de TF configurado."""
        if self.tf_scheme == "raw":
            return float(raw_tf)
        if self.tf_scheme == "log":
            # Log-TF suaviza documentos que repetem muito o mesmo termo
            return 1.0 + math.log(raw_tf) if raw_tf > 0 else 0.0
        if self.tf_scheme == "boolean":
            return 1.0 if raw_tf > 0 else 0.0
        return float(raw_tf)

    @staticmethod
    def _compute_idf(token_lists: dict, N: int) -> dict:
        """
        Calcula IDF com suavização para evitar divisão por zero e
        para dar algum peso a termos que aparecem em todos os documentos.

        Fórmula: idf(t) = log((N+1) / (df(t)+1)) + 1
        """
        df: dict = {}
        for tokens in token_lists.values():
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        idf: dict = {}
        for term, count in df.items():
            idf[term] = math.log((N + 1) / (count + 1)) + 1.0
        return idf

    @staticmethod
    def _cosine_similarity(v1: dict, v2: dict) -> float:
        """
        Similaridade de cosseno entre dois vetores esparsos.
        Só percorre os termos em comum — muito mais eficiente
        do que trabalhar com vetores densos de dimensão |vocabulário|.
        """
        common = set(v1.keys()) & set(v2.keys())
        if not common:
            return 0.0
        dot = sum(v1[t] * v2[t] for t in common)
        norm1 = math.sqrt(sum(w * w for w in v1.values()))
        norm2 = math.sqrt(sum(w * w for w in v2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    # ------------------------------------------------------------------
    # Backend sklearn
    # ------------------------------------------------------------------

    def _build_sklearn(self, documents: list) -> None:
        """Constrói o índice sklearn: transforma documentos em matriz TF-IDF esparsa."""
        self._sklearn_doc_ids = list(self._documents.keys())
        corpus = []
        for doc_id in self._sklearn_doc_ids:
            doc = self._documents[doc_id]
            tokens = self.preprocessor.process_document(doc, self.fields)
            corpus.append(" ".join(tokens))
        self._sklearn_vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"\S+",
            sublinear_tf=(self.tf_scheme == "log"),
        )
        self._sklearn_matrix = self._sklearn_vectorizer.fit_transform(corpus)

    def _search_sklearn(self, query: str, top_k: int) -> list:
        """Pesquisa usando o vectorizer sklearn já treinado."""
        if self._sklearn_vectorizer is None:
            return []
        query_tokens = self.preprocessor.process(query)
        query_str = " ".join(query_tokens)
        try:
            query_vec = self._sklearn_vectorizer.transform([query_str])
        except Exception as exc:
            logger.error("Erro ao transformar query com sklearn: %s", exc)
            return []
        scores = sklearn_cosine(query_vec, self._sklearn_matrix).flatten()
        results = []
        for idx, score in enumerate(scores):
            if score > 0:
                doc_id = self._sklearn_doc_ids[idx]
                results.append(
                    TFIDFResult(doc_id=doc_id, document=self._documents[doc_id], score=round(float(score), 6))
                )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Similaridade entre documentos
    # ------------------------------------------------------------------

    def similarity_matrix(self) -> tuple[np.ndarray, list[int]]:
        """
        Calcula a matriz N×N de similaridades entre todos os documentos.

        A diagonal é sempre 1.0 (um documento é idêntico a si próprio).
        Funciona com ambos os backends.

        Returns
        -------
        matrix : np.ndarray de forma (N, N)
        doc_ids : lista de IDs correspondente a cada linha/coluna
        """
        if not self._documents:
            return np.empty((0, 0)), []

        if self.use_sklearn and self._sklearn_matrix is not None:
            doc_ids = list(self._sklearn_doc_ids)
            matrix = sklearn_cosine(self._sklearn_matrix, self._sklearn_matrix)
            return matrix.astype(float), doc_ids

        # Backend custom — construído a partir dos vetores esparsos
        doc_ids = sorted(self._doc_vectors.keys())
        n = len(doc_ids)
        matrix = np.zeros((n, n), dtype=float)

        for i, id_i in enumerate(doc_ids):
            matrix[i][i] = 1.0
            for j in range(i + 1, n):
                id_j = doc_ids[j]
                sim = self._cosine_similarity(self._doc_vectors[id_i], self._doc_vectors[id_j])
                matrix[i][j] = sim
                matrix[j][i] = sim   # a matriz é simétrica

        logger.info("Matriz de similaridade calculada: %dx%d", n, n)
        return matrix, doc_ids

    def similar_to(self, doc_id: int, top_k: int = 10) -> list:
        """
        Devolve os top-K documentos mais similares ao documento dado.

        Usa a mesma lógica da similarity_matrix mas sem computar a
        matriz completa quando estamos no backend custom.

        Parameters
        ----------
        doc_id : int  — documento de referência
        top_k  : int  — quantos resultados devolver (excluindo o próprio)
        """
        if doc_id not in self._documents:
            return []

        if self.use_sklearn and self._sklearn_matrix is not None:
            try:
                idx = self._sklearn_doc_ids.index(doc_id)
            except ValueError:
                return []
            query_vec = self._sklearn_matrix[idx]
            scores = sklearn_cosine(query_vec, self._sklearn_matrix).flatten()
            results = []
            for i, score in enumerate(scores):
                other_id = self._sklearn_doc_ids[i]
                if other_id != doc_id and score > 0:
                    results.append(SimilarityResult(
                        doc_id=other_id,
                        document=self._documents[other_id],
                        similarity=round(float(score), 6),
                    ))
        else:
            if doc_id not in self._doc_vectors:
                return []
            ref_vector = self._doc_vectors[doc_id]
            results = []
            for other_id, other_vector in self._doc_vectors.items():
                if other_id == doc_id:
                    continue
                sim = self._cosine_similarity(ref_vector, other_vector)
                if sim > 0:
                    results.append(SimilarityResult(
                        doc_id=other_id,
                        document=self._documents[other_id],
                        similarity=round(sim, 6),
                    ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    @property
    def num_documents(self) -> int:
        return len(self._documents)

    def get_document(self, doc_id: int):
        return self._documents.get(doc_id)

    def get_term_idf(self, term: str) -> float:
        """Devolve o IDF de um termo (após pré-processamento)."""
        processed = self.preprocessor.process(term)
        if not processed:
            return 0.0
        return self._idf.get(processed[0], 0.0)

    def stats(self) -> dict:
        """Resumo do estado atual do índice."""
        top_idf = sorted(self._idf.items(), key=lambda x: x[1], reverse=True)[:10]
        return {
            "num_documents": self.num_documents,
            "vocabulary_size": len(self._vocabulary),
            "use_sklearn": self.use_sklearn,
            "tf_scheme": self.tf_scheme,
            "top_idf_terms": top_idf,  # termos mais raros (maior IDF)
        }

    def __repr__(self) -> str:
        backend = "sklearn" if self.use_sklearn else "custom"
        return f"TFIDFEngine(docs={self.num_documents}, terms={len(self._vocabulary)}, backend={backend})"
