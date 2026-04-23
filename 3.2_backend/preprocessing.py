# preprocessing.py

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize


class Preprocessor:
    def __init__(
        self,
        language="pt",
        remove_stopwords=True,
        use_stemming=True,
        use_lemmatization=False
    ):
        """
        Preprocessor for text normalization.
        Supports PT/EN, stopwords, stemming, lemmatization.

        Args:
            language (str): "pt" or "en"
            remove_stopwords (bool)
            use_stemming (bool)
            use_lemmatization (bool)
        """

        self.language = language
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization

        # Load stopwords
        try:
            self.stopwords = set(stopwords.words("portuguese" if language == "pt" else "english"))
        except:
            self.stopwords = set()

        # Stemmer (Porter)
        self.stemmer = PorterStemmer()

        # Lemmatizer (WordNet)
        self.lemmatizer = WordNetLemmatizer()

    # ---------------------------------------------------------
    # Sentence segmentation
    # ---------------------------------------------------------
    def sentence_split(self, text):
        return sent_tokenize(text)

    # ---------------------------------------------------------
    # Tokenization
    # ---------------------------------------------------------
    def tokenize(self, text):
        # Remove non-alphanumeric chars (keep accents)
        text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text)
        tokens = word_tokenize(text, language="portuguese" if self.language == "pt" else "english")
        return tokens

    # ---------------------------------------------------------
    # Stemming
    # ---------------------------------------------------------
    def stem(self, token):
        return self.stemmer.stem(token)

    # ---------------------------------------------------------
    # Lemmatization
    # ---------------------------------------------------------
    def lemmatize(self, token):
        # WordNet works only for English
        if self.language == "en":
            return self.lemmatizer.lemmatize(token)
        return token  # fallback for PT

    # ---------------------------------------------------------
    # Full preprocessing pipeline
    # ---------------------------------------------------------
    def preprocess(self, text):
        """
        Applies:
        - lowercase
        - tokenization
        - stopword removal (optional)
        - stemming (optional)
        - lemmatization (optional)
        """

        # Lowercase
        text = text.lower()

        # Tokenize
        tokens = self.tokenize(text)

        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        # Stemming
        if self.use_stemming:
            tokens = [self.stem(t) for t in tokens]

        # Lemmatization
        if self.use_lemmatization:
            tokens = [self.lemmatize(t) for t in tokens]

        # Remove empty tokens
        tokens = [t for t in tokens if len(t) > 0]

        return tokens
