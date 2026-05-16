"""
inverted_index.py — Índice invertido com listas de postings.

É a estrutura central do motor de pesquisa booleana. A ideia é simples:
em vez de percorrer todos os documentos para cada query, mantemos um
dicionário que mapeia cada termo para a lista de documentos onde aparece.

Adicionalmente guardamos as posições de cada ocorrência, o que permite
pesquisas de proximidade (ex: dois termos a menos de N palavras de distância).

Os skip pointers aceleram a interseção de listas longas em queries AND —
em vez de comparar elemento a elemento, podemos "saltar" blocos quando
sabemos que nenhum elemento nesse bloco pode fazer match.
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
# Estruturas de dados
# ---------------------------------------------------------------------------

@dataclass
class Posting:
    """
    Registo de uma ocorrência de um termo num documento.

    Attributes:
        doc_id:    ID único do documento.
        tf:        Número total de ocorrências do termo neste documento.
        positions: Lista de posições (offset em tokens) onde o termo aparece.
                   Usado em pesquisas de frase e proximidade.
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
    Lista de postings de um termo, com metadados e skip pointers.

    Attributes:
        df:       Document frequency — em quantos documentos este termo aparece.
        postings: Lista de Postings, ordenada por doc_id (obrigatório para skip pointers).
        skips:    Dicionário {posição_i: posição_j} para saltos na interseção.
    """
    df: int = 0
    postings: list[Posting] = field(default_factory=list)
    skips: dict[int, int] = field(default_factory=dict)

    def build_skip_pointers(self) -> None:
        """
        Constrói skip pointers com intervalo sqrt(n).

        Para listas com menos de 3 elementos não vale a pena —
        o overhead de verificar o skip seria maior que a poupança.
        """
        n = len(self.postings)
        if n < 3:
            self.skips = {}
            return

        skip_interval = max(1, int(math.sqrt(n)))
        self.skips = {}
        for i in range(0, n - skip_interval, skip_interval):
            self.skips[i] = i + skip_interval

    def to_dict(self) -> dict:
        return {
            "df": self.df,
            "postings": [p.to_dict() for p in self.postings],
            "skips": {str(k): v for k, v in self.skips.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> "PostingsList":
        return PostingsList(
            df=d["df"],
            postings=[Posting.from_dict(p) for p in d["postings"]],
            skips={int(k): v for k, v in d.get("skips", {}).items()},
        )


# ---------------------------------------------------------------------------
# Índice invertido principal
# ---------------------------------------------------------------------------

class InvertedIndex:
    """
    Índice invertido com skip pointers e suporte a atualizações incrementais.

    Uso básico:
        pp = make_stemming_preprocessor()
        idx = InvertedIndex(pp)
        idx.build_from_documents(list_of_docs)
        postings = idx.get_postings("information")
    """

    def __init__(
        self,
        preprocessor: Optional[Preprocessor] = None,
        fields: Optional[list[str]] = None,
    ):
        self.preprocessor = preprocessor or make_stemming_preprocessor("english")
        self.fields = fields or ["title", "abstract"]

        self._index: dict[str, PostingsList] = {}
        self._documents: dict[int, dict] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------

    @property
    def num_documents(self) -> int:
        return len(self._documents)

    @property
    def vocabulary_size(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    # Construção do índice
    # ------------------------------------------------------------------

    def build_from_documents(self, documents: list[dict]) -> None:
        """
        Reconstrói o índice a partir de uma lista de documentos.
        Limpa qualquer estado anterior.
        """
        self._index.clear()
        self._documents.clear()
        self._next_id = 0

        for doc in documents:
            self._index_document(doc)

        # Skip pointers são construídos no fim, quando as listas já estão completas
        self._rebuild_skip_pointers()

    def add_document(self, doc: dict) -> int:
        """
        Adiciona um único documento ao índice existente (atualização incremental).
        Atualiza os skip pointers apenas nas listas afetadas.

        Returns:
            O doc_id atribuído ao documento adicionado.
        """
        doc_id = self._index_document(doc)

        tokens = set(self.preprocessor.process_document(doc, self.fields))
        for term in tokens:
            pl = self._index.get(term)
            if pl:
                self._update_skip_incremental(pl)

        logger.debug("Documento %d adicionado ao índice.", doc_id)
        return doc_id

    # ------------------------------------------------------------------
    # Indexação interna
    # ------------------------------------------------------------------

    def _index_document(self, doc: dict) -> int:
        """
        Indexa um documento: atribui um ID, extrai tokens por campo
        e atualiza as listas de postings correspondentes.
        """
        doc_id = self._next_id
        self._documents[doc_id] = doc
        self._next_id += 1

        position = 0
        term_data: dict[str, Posting] = {}

        for fname in self.fields:
            value = doc.get(fname, "")
            if isinstance(value, list):
                value = " ".join(value)
            if not isinstance(value, str) or not value.strip():
                continue

            tokens = self.preprocessor.process(value)
            for token in tokens:
                if token not in term_data:
                    term_data[token] = Posting(doc_id=doc_id, tf=0, positions=[])
                term_data[token].tf += 1
                term_data[token].positions.append(position)
                position += 1

        # Adicionar os postings ao índice global
        for term, posting in term_data.items():
            if term not in self._index:
                self._index[term] = PostingsList(df=0, postings=[])
            pl = self._index[term]
            pl.postings.append(posting)
            pl.df += 1

        # Garantir que os postings ficam ordenados por doc_id —
        # é obrigatório para que os skip pointers funcionem corretamente
        for pl in self._index.values():
            pl.postings.sort(key=lambda p: p.doc_id)

        return doc_id

    # ------------------------------------------------------------------
    # Skip pointers
    # ------------------------------------------------------------------

    def _rebuild_skip_pointers(self) -> None:
        """Reconstrói os skip pointers para todas as listas do índice."""
        for pl in self._index.values():
            pl.build_skip_pointers()

    def _update_skip_incremental(self, pl: PostingsList) -> None:
        """
        Atualiza os skip pointers de uma lista individual.
        Chamado após adicionar um documento individualmente.
        """
        n = len(pl.postings)
        if n < 3:
            pl.skips = {}
            return

        skip_interval = max(1, int(math.sqrt(n)))
        pl.skips = {}
        for i in range(0, n - skip_interval, skip_interval):
            pl.skips[i] = i + skip_interval

    # ------------------------------------------------------------------
    # Operações de query
    # ------------------------------------------------------------------

    def get_postings(self, term: str) -> list[Posting]:
        """
        Devolve a lista de postings de um termo, aplicando pré-processamento.
        Este é o método que se deve usar em queries normais.
        """
        processed = self.preprocessor.process(term)
        if not processed:
            return []
        pl = self._index.get(processed[0])
        return pl.postings if pl else []

    def get_postings_raw(self, term: str) -> list[Posting]:
        """
        Devolve a lista de postings sem pré-processamento.
        Útil quando o termo já foi processado pelo caller.
        """
        pl = self._index.get(term)
        return pl.postings if pl else []

    def intersect(self, term1: str, term2: str) -> list[Posting]:
        """
        Interseção de postings de dois termos (AND booleano).
        Aplica pré-processamento a ambos os termos.
        """
        t1 = self.preprocessor.process(term1)
        t2 = self.preprocessor.process(term2)

        if not t1 or not t2:
            return []

        pl1 = self._index.get(t1[0])
        pl2 = self._index.get(t2[0])

        if not pl1 or not pl2:
            return []

        return self._intersect_postings(pl1, pl2)

    # ------------------------------------------------------------------
    # Interseção com skip pointers
    # ------------------------------------------------------------------

    def _intersect_postings(self, pl1: PostingsList, pl2: PostingsList) -> list[Posting]:
        """
        Merge-join das duas listas de postings, usando skip pointers
        para saltar blocos quando possível.

        O algoritmo avança os dois ponteiros em paralelo:
        - Se doc_ids iguais → match, avança ambos.
        - Se doc1 < doc2 → tenta usar skip em pl1 para chegar mais perto de doc2.
        - Se doc1 > doc2 → idem para pl2.
        """
        result = []

        p1 = pl1.postings
        p2 = pl2.postings
        skips1 = pl1.skips
        skips2 = pl2.skips

        i, j = 0, 0

        while i < len(p1) and j < len(p2):
            doc1 = p1[i].doc_id
            doc2 = p2[j].doc_id

            if doc1 == doc2:
                result.append(p1[i])
                i += 1
                j += 1

            elif doc1 < doc2:
                # Tentar saltar em pl1
                if i in skips1:
                    skip_i = skips1[i]
                    if skip_i < len(p1) and p1[skip_i].doc_id <= doc2:
                        i = skip_i
                    else:
                        i += 1
                else:
                    i += 1

            else:
                # Tentar saltar em pl2
                if j in skips2:
                    skip_j = skips2[j]
                    if skip_j < len(p2) and p2[skip_j].doc_id <= doc1:
                        j = skip_j
                    else:
                        j += 1
                else:
                    j += 1

        return result

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "num_documents": self.num_documents,
            "vocabulary_size": self.vocabulary_size,
        }

    def __repr__(self) -> str:
        return f"InvertedIndex(docs={self.num_documents}, terms={self.vocabulary_size})"
