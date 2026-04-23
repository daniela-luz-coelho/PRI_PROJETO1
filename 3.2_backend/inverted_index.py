# inverted_index.py

import json
from preprocessing import Preprocessor


class InvertedIndexBuilder:
    def __init__(self, language="pt"):
        self.preprocessor = Preprocessor(language=language)
        self.index = {}          # termo → lista de docs
        self.doc_ids = []        # lista de IDs dos documentos
        self.total_docs = 0

    # ---------------------------------------------------------
    # 1) Ler o ficheiro JSON do scraper
    # ---------------------------------------------------------
    def load_raw_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    # ---------------------------------------------------------
    # 2) Construir o índice invertido
    # ---------------------------------------------------------
    def build_index(self, raw_data):
        """
        raw_data: lista de documentos do scraper
        Cada documento deve ter pelo menos: "title" e "abstract"
        """
        self.index = {}
        self.doc_ids = list(range(1, len(raw_data) + 1))
        self.total_docs = len(raw_data)

        for doc_id, doc in zip(self.doc_ids, raw_data):
            # Extrair texto relevante
            text = f"{doc.get('title', '')} {doc.get('abstract', '')}"

            # Pré-processar
            tokens = self.preprocessor.preprocess(text)

            # Inserir no índice
            for token in tokens:
                if token not in self.index:
                    self.index[token] = []
                if doc_id not in self.index[token]:
                    self.index[token].append(doc_id)

        # Ordenar postings lists
        for term in self.index:
            self.index[term].sort()

        return self.index

    # ---------------------------------------------------------
    # 3) Guardar o índice em ficheiro
    # ---------------------------------------------------------
    def save_index(self, path):
        data = {
            "index": self.index,
            "total_docs": self.total_docs,
            "doc_ids": self.doc_ids
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    # ---------------------------------------------------------
    # 4) Carregar índice de ficheiro
    # ---------------------------------------------------------
    @staticmethod
    def load_index(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["index"], data["total_docs"], data["doc_ids"]
