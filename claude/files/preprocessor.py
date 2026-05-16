"""
preprocessor.py — Pipeline de pré-processamento de texto.

Trata do trabalho chato mas essencial de transformar texto bruto
em tokens prontos a indexar. A ordem das operações importa:
primeiro limpa-se, depois filtra-se, depois normaliza-se a raiz.

Uso básico:
    pp = Preprocessor(language="english", use_stemming=True)
    tokens = pp.process("Information retrieval systems are useful.")
    # → ['inform', 'retriev', 'system', 'use']

Para usar a configuração padrão do projeto, as factory functions
no fim do ficheiro poupam algumas linhas:
    pp = make_stemming_preprocessor()
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

import nltk
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Garantir que os recursos NLTK necessários estão disponíveis
# ---------------------------------------------------------------------------

_NLTK_PACKAGES = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]


def _download_nltk_resources() -> None:
    """Faz download silencioso de recursos NLTK que ainda não existam localmente."""
    for pkg in _NLTK_PACKAGES:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            pass  # será descarregado a seguir

    for pkg in _NLTK_PACKAGES:
        nltk.download(pkg, quiet=True)


_download_nltk_resources()

# ---------------------------------------------------------------------------
# Tipo para a linguagem suportada
# ---------------------------------------------------------------------------

Language = Literal["english", "portuguese"]

# ---------------------------------------------------------------------------
# Mapeamento de POS tags (Penn Treebank → WordNet)
# ---------------------------------------------------------------------------

def _pos_to_wordnet(treebank_tag: str) -> str:
    """
    Converte uma etiqueta POS do Penn Treebank para a constante WordNet.
    Necessário para que o lematizador saiba se 'running' é verbo ou nome.
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    # Por defeito assume-se nome — é a categoria mais comum
    return wordnet.NOUN


# ---------------------------------------------------------------------------
# Configuração do pipeline
# ---------------------------------------------------------------------------

@dataclass
class PreprocessorConfig:
    """
    Controla o que cada instância do Preprocessor faz.

    Campos:
        language:           Língua principal para stopwords e stemming.
        use_stemming:       Aplica Porter/Snowball se True.
        use_lemmatisation:  Aplica WordNet lemmatizer se True.
                            Se ambos estiverem a True, a lematização corre
                            primeiro e o stemming depois.
        remove_stopwords:   Filtra stopwords se True.
        extra_stopwords:    Termos adicionais a remover (específicos do domínio).
        min_token_length:   Descarta tokens mais curtos que este valor.
        lowercase:          Converte tudo para minúsculas.
        remove_punctuation: Remove tokens sem caracteres alfanuméricos.
        remove_numbers:     Remove tokens exclusivamente numéricos.
    """
    language: Language = "english"
    use_stemming: bool = False
    use_lemmatisation: bool = True
    remove_stopwords: bool = True
    extra_stopwords: list[str] = field(default_factory=list)
    min_token_length: int = 2
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    Pipeline NLP configurável para pré-processamento de texto.

    Exemplo com português:
        pp = Preprocessor(language="portuguese", use_stemming=True)
        pp.process("Os sistemas de recuperação de informação são úteis.")
        # → ['sistem', 'recuper', 'inform', 'útei']
    """

    def __init__(self, config: PreprocessorConfig | None = None, **kwargs):
        if config is None:
            config = PreprocessorConfig(**kwargs)
        self.config = config

        # --- Stopwords ---
        self._stopwords: set[str] = set()
        if config.remove_stopwords:
            self._stopwords = self._build_stopwords(config.language)
        if config.extra_stopwords:
            self._stopwords.update(w.lower() for w in config.extra_stopwords)

        # --- Stemmer (escolhido por língua) ---
        self._stemmer = None
        if config.use_stemming:
            if config.language == "portuguese":
                self._stemmer = SnowballStemmer("portuguese")
            else:
                self._stemmer = PorterStemmer()

        # --- Lematizador (apenas para inglês funciona bem com POS) ---
        self._lemmatiser: WordNetLemmatizer | None = None
        if config.use_lemmatisation:
            self._lemmatiser = WordNetLemmatizer()

        logger.info(
            "Preprocessor pronto — lang=%s | stemming=%s | lemmatização=%s | stopwords=%s",
            config.language,
            config.use_stemming,
            config.use_lemmatisation,
            config.remove_stopwords,
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def process(self, text: str) -> list[str]:
        """
        Corre o pipeline completo sobre o texto recebido.

        Etapas (por ordem):
            1. Tokenização
            2. Lowercase
            3. Remoção de pontuação / números
            4. Remoção de stopwords
            5. Lematização
            6. Stemming
            7. Filtro de comprimento mínimo

        Args:
            text: Texto bruto de entrada.

        Returns:
            Lista de tokens processados.
        """
        if not text or not text.strip():
            return []

        tokens = self._tokenise(text)

        if self.config.lowercase:
            tokens = [t.lower() for t in tokens]

        if self.config.remove_punctuation:
            tokens = [t for t in tokens if re.search(r"\w", t)]

        if self.config.remove_numbers:
            tokens = [t for t in tokens if not t.isdigit()]

        if self.config.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]

        if self._lemmatiser:
            tokens = self._lemmatise(tokens)

        if self._stemmer:
            tokens = [self._stemmer.stem(t) for t in tokens]

        tokens = [t for t in tokens if len(t) >= self.config.min_token_length]

        return tokens

    def process_document(self, doc: dict, fields: list[str] | None = None) -> list[str]:
        """
        Processa um dicionário de publicação e devolve todos os tokens concatenados.

        Args:
            doc:    Dicionário com metadados da publicação (formato do scraper).
            fields: Campos a processar. Por defeito: title + abstract.

        Returns:
            Lista flat de tokens de todos os campos pedidos.
        """
        if fields is None:
            fields = ["title", "abstract"]

        tokens: list[str] = []
        for field_name in fields:
            value = doc.get(field_name, "")
            if isinstance(value, list):      # ex: lista de autores
                value = " ".join(value)
            if isinstance(value, str) and value.strip() and value != "N/A":
                tokens.extend(self.process(value))
        return tokens

    def segment_sentences(self, text: str) -> list[str]:
        """
        Divide o texto em frases usando o tokenizador de frases do NLTK.

        Args:
            text: Texto de entrada.

        Returns:
            Lista de frases.
        """
        lang = "portuguese" if self.config.language == "portuguese" else "english"
        return sent_tokenize(text, language=lang)

    def get_stopwords(self) -> set[str]:
        """Devolve cópia do conjunto de stopwords em uso."""
        return set(self._stopwords)

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _tokenise(self, text: str) -> list[str]:
        """Tokeniza o texto usando o word_tokenize do NLTK."""
        lang = "portuguese" if self.config.language == "portuguese" else "english"
        return word_tokenize(text, language=lang)

    def _lemmatise(self, tokens: list[str]) -> list[str]:
        """
        Lematização com consciência da categoria gramatical (apenas inglês).
        Para português, lematiza como nomes por ser a melhor opção disponível
        sem instalar modelos externos.
        """
        if self.config.language == "english":
            pos_tags = nltk.pos_tag(tokens)
            return [
                self._lemmatiser.lemmatize(token, _pos_to_wordnet(tag))
                for token, tag in pos_tags
            ]
        return [self._lemmatiser.lemmatize(t) for t in tokens]

    @staticmethod
    def _build_stopwords(language: Language) -> set[str]:
        """
        Constrói um conjunto de stopwords combinando inglês e português.
        Incluímos sempre as duas línguas porque o corpus tem documentos em ambas.
        """
        words: set[str] = set()
        for lang in ("english", "portuguese"):
            try:
                words.update(stopwords.words(lang))
            except OSError:
                logger.warning("Stopwords para '%s' não disponíveis.", lang)
        return words

    @staticmethod
    def normalise_unicode(text: str) -> str:
        """
        Remove acentuação via decomposição NFD.
        Útil antes do stemming para normalizar variantes acentuadas.

        Exemplo: "recuperação" → "recuperacao"
        """
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )


# ---------------------------------------------------------------------------
# Factory functions — para não ter de instanciar config a mão em todo o lado
# ---------------------------------------------------------------------------

def make_stemming_preprocessor(language: Language = "english") -> Preprocessor:
    """Preprocessor com stemming, sem lematização. Mais rápido."""
    return Preprocessor(
        PreprocessorConfig(
            language=language,
            use_stemming=True,
            use_lemmatisation=False,
            remove_stopwords=True,
        )
    )


def make_lemmatisation_preprocessor(language: Language = "english") -> Preprocessor:
    """Preprocessor com lematização, sem stemming. Mais preciso."""
    return Preprocessor(
        PreprocessorConfig(
            language=language,
            use_stemming=False,
            use_lemmatisation=True,
            remove_stopwords=True,
        )
    )


def make_bare_preprocessor(language: Language = "english") -> Preprocessor:
    """Preprocessor mínimo: só tokeniza e converte para minúsculas."""
    return Preprocessor(
        PreprocessorConfig(
            language=language,
            use_stemming=False,
            use_lemmatisation=False,
            remove_stopwords=False,
        )
    )
