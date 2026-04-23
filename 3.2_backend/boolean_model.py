# boolean_model.py

from preprocessing import Preprocessor


class BooleanModel:
    def __init__(self, inverted_index, total_docs, language="pt"):
        """
        inverted_index: dict {term: [doc_ids]}
        total_docs: número total de documentos
        """
        self.index = inverted_index
        self.total_docs = total_docs
        self.preprocessor = Preprocessor(language=language)

    # ---------------------------------------------------------
    # 1) Pré-processamento da query
    # ---------------------------------------------------------
    def preprocess_query(self, query):
        tokens = self.preprocessor.preprocess(query)
        return tokens

    # ---------------------------------------------------------
    # 2) Inserir AND implícito
    # Ex: ["cancer", "breast"] → ["cancer", "AND", "breast"]
    # ---------------------------------------------------------
    def insert_implicit_and(self, tokens):
        result = []
        for i, tok in enumerate(tokens):
            result.append(tok)
            if i < len(tokens) - 1:
                next_tok = tokens[i + 1]
                if tok not in ["and", "or", "not"] and next_tok not in ["and", "or", "not"]:
                    result.append("and")
        return result

    # ---------------------------------------------------------
    # 3) Avaliar a expressão booleana (versão simples)
    # Precedência: NOT > AND > OR
    # ---------------------------------------------------------
    def evaluate(self, tokens):
        # -----------------------------------------------------
        # 3.1 — Resolver NOT
        # -----------------------------------------------------
        i = 0
        while i < len(tokens):
            if tokens[i] == "not":
                term = tokens[i + 1]

                # Caso 1: já é uma lista de docs
                if isinstance(term, list):
                    docs_with_term = set(term)
                else:
                    # Caso 2: é string → ir buscar ao índice
                    docs_with_term = set(self.index.get(term, []))

                all_docs = set(range(1, self.total_docs + 1))
                result = sorted(all_docs - docs_with_term)

                tokens[i:i+2] = [result]  # substituir NOT + termo por lista
            else:
                i += 1

        # -----------------------------------------------------
        # 3.2 — Resolver AND
        # -----------------------------------------------------
        i = 0
        while i < len(tokens):
            if tokens[i] == "and":
                left = tokens[i - 1]
                right = tokens[i + 1]

                result = self.op_and(left, right)
                tokens[i-1:i+2] = [result]
            else:
                i += 1

        # -----------------------------------------------------
        # 3.3 — Resolver OR
        # -----------------------------------------------------
        i = 0
        while i < len(tokens):
            if tokens[i] == "or":
                left = tokens[i - 1]
                right = tokens[i + 1]

                result = self.op_or(left, right)
                tokens[i-1:i+2] = [result]
            else:
                i += 1

        # No fim sobra uma lista com 1 elemento: a lista final de docs
        return tokens[0]

    # ---------------------------------------------------------
    # Operações booleanas
    # ---------------------------------------------------------
    def op_and(self, left, right):
        return sorted(set(left) & set(right))

    def op_or(self, left, right):
        return sorted(set(left) | set(right))

    # ---------------------------------------------------------
    # Obter lista de docs para um termo
    # ---------------------------------------------------------
    def get_postings(self, term):
        return self.index.get(term, [])

    # ---------------------------------------------------------
    # Função principal de pesquisa
    # ---------------------------------------------------------
    def search(self, query):
        tokens = self.preprocess_query(query)
        tokens = self.insert_implicit_and(tokens)

        # Substituir termos por listas de docs
        processed = []
        for tok in tokens:
            if tok in ["and", "or", "not"]:
                processed.append(tok)
            else:
                processed.append(self.get_postings(tok))

        return self.evaluate(processed)
