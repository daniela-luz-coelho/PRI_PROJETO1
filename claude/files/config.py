"""
config.py — Configuração central do motor de pesquisa.

Em vez de andar a mudar valores espalhados por vários ficheiros,
centralizámos tudo aqui. Para usar noutro módulo basta fazer:

    from src.config import settings
    print(settings.TOP_K_DEFAULT)

Se precisares de mudar algum valor em testes ou deployment sem tocar
no código, podes passar variáveis de ambiente com o prefixo IR_:

    IR_API_PORT=9000 python -m uvicorn ...
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default):
    """Lê uma variável de ambiente e converte para o mesmo tipo do default."""
    raw = os.environ.get(f"IR_{key}")
    if raw is None:
        return default
    try:
        return type(default)(raw)
    except (ValueError, TypeError):
        return default


@dataclass
class Settings:
    # --- Caminhos base ---
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_FILE: Path = BASE_DIR / "src" / "scraper" / "scraper_results.json"
    DB_PATH: str = "data/repositorium.db"
    INDEX_CACHE_DIR: str = "data/index_cache"

    # --- Scraper ---
    # Quantos documentos recolher no máximo. 500 é um bom equilíbrio
    # entre ter dados suficientes e não demorar horas a fazer scraping.
    SCRAPER_MAX_ITEMS: int = 500
    SCRAPER_DEFAULT_AREA: str = "computer_science"
    SCRAPER_COLLECTIONS: list = field(default_factory=lambda: [
        "1822/21293",  # Informática / Ciências da Computação
        "1822/68418",  # Ciências da Saúde
    ])

    # --- Pré-processamento ---
    # Stemming é mais rápido que lematização mas produz radicais
    # que por vezes não são palavras reais. Ficámos com stemming
    # como default por ser mais leve em produção.
    DEFAULT_PREPROCESSING: str = "stemming"  # "stemming" | "lemmatisation" | "bare"
    DEFAULT_LANGUAGE: str = "english"
    REMOVE_STOPWORDS: bool = True
    MIN_TOKEN_LENGTH: int = 2

    # --- TF-IDF ---
    # Log-TF suaviza o efeito de documentos que repetem muito um termo
    TF_SCHEME: str = "log"  # "log" | "raw" | "boolean"
    DEFAULT_TFIDF_BACKEND: str = "custom"  # "custom" | "sklearn"
    TFIDF_FIELDS: list = field(default_factory=lambda: ["title", "abstract"])

    # --- API de pesquisa ---
    TOP_K_DEFAULT: int = 20
    TOP_K_MAX: int = 100
    # Janela em torno da ocorrência do termo para gerar snippets
    SNIPPET_WINDOW_CHARS: int = 120
    SNIPPET_MAX_LENGTH: int = 300
    # Máximo de sinónimos a adicionar por termo na expansão de query
    QUERY_EXPANSION_MAX_SYNONYMS: int = 2
    PHRASE_QUERY_PROXIMITY_WINDOW: int = 3

    # --- Classificador ---
    CLASSIFIER_MAX_FEATURES: int = 5000
    CLASSIFIER_ALPHA: float = 1.0         # suavização Laplace
    CLASSIFIER_RUN_CROSS_VALIDATION: bool = False  # True é mais lento

    # --- Desempenho ---
    BATCH_SIZE: int = 100

    # --- Servidor ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = False  # True apenas em desenvolvimento (recarrega ao gravar)
    CORS_ORIGINS: list = field(default_factory=lambda: ["*"])
    LOG_LEVEL: str = "INFO"

    def __post_init__(self):
        """Aplica overrides das variáveis de ambiente (se existirem)."""
        self.TOP_K_DEFAULT  = _env("TOP_K_DEFAULT",  self.TOP_K_DEFAULT)
        self.TOP_K_MAX      = _env("TOP_K_MAX",       self.TOP_K_MAX)
        self.API_PORT       = _env("API_PORT",        self.API_PORT)
        self.API_HOST       = _env("API_HOST",        self.API_HOST)
        self.API_RELOAD     = _env("API_RELOAD",      self.API_RELOAD)
        self.DB_PATH        = _env("DB_PATH",         self.DB_PATH)
        self.LOG_LEVEL      = _env("LOG_LEVEL",       self.LOG_LEVEL)
        self.BATCH_SIZE     = _env("BATCH_SIZE",      self.BATCH_SIZE)
        self.DEFAULT_PREPROCESSING = _env("DEFAULT_PREPROCESSING", self.DEFAULT_PREPROCESSING)
        self.REMOVE_STOPWORDS      = _env("REMOVE_STOPWORDS",      self.REMOVE_STOPWORDS)
        self.TF_SCHEME             = _env("TF_SCHEME",             self.TF_SCHEME)
        self.DEFAULT_TFIDF_BACKEND = _env("DEFAULT_TFIDF_BACKEND", self.DEFAULT_TFIDF_BACKEND)
        self.SNIPPET_WINDOW_CHARS  = _env("SNIPPET_WINDOW_CHARS",  self.SNIPPET_WINDOW_CHARS)
        self.QUERY_EXPANSION_MAX_SYNONYMS = _env(
            "QUERY_EXPANSION_MAX_SYNONYMS", self.QUERY_EXPANSION_MAX_SYNONYMS
        )


# Singleton — importar sempre este objeto, nunca instanciar Settings diretamente
settings = Settings()
