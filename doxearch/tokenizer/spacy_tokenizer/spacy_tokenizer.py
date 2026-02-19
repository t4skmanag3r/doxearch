import spacy
from snowballstemmer import stemmer

from doxearch.tokenizer.tokenizer import Tokenizer

# Map spaCy model language codes to snowballstemmer language names
stemmer_language_map = {
    "ar": "arabic",
    "hy": "armenian",
    "eu": "basque",
    "ca": "catalan",
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "eo": "esperanto",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "el": "greek",
    "hi": "hindi",
    "hu": "hungarian",
    "id": "indonesian",
    "ga": "irish",
    "it": "italian",
    "lt": "lithuanian",
    "ne": "nepali",
    "nb": "norwegian",
    "no": "norwegian",
    "nn": "norwegian",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sr": "serbian",
    "es": "spanish",
    "sv": "swedish",
    "ta": "tamil",
    "tr": "turkish",
    "yi": "yiddish",
}

supported_stemmer_languages = list(stemmer_language_map.values())


class SpacyTokenizer(Tokenizer):

    def __init__(
        self,
        model: str = "en_core_web_sm",
        use_lemmatization: bool = True,
        use_stemming: bool = False,
        disable: list[str] | None = None,
    ):
        """Initialize SpacyTokenizer with a spacy language model.

        Args:
            model: Name of the spacy model to use (default: "en_core_web_sm")
            use_lemmatization: Whether to use lemmatization for word normalization
            use_stemming: Whether to use stemming (only for Lithuanian)
            disable: List of pipeline components to disable for faster processing.
                    Default disables parser and NER, keeping only tokenizer.
        """
        self.model_name = model
        self.use_lemmatization = use_lemmatization
        self.use_stemming = use_stemming

        # Detect language from model name or use provided language
        detected_language = self._detect_language_from_model(model)

        # Initialize stemmer if stemming is enabled
        if self.use_stemming:
            if detected_language:
                try:
                    self.stemmer = stemmer(detected_language)
                except KeyError:
                    raise ValueError(
                        f"Stemmer for language '{detected_language}' not supported. "
                        f"Supported languages: {supported_stemmer_languages}"
                    )
            else:
                raise ValueError(
                    f"Could not detect stemmer language from model '{model}'. "
                    f"Please specify stemmer_language parameter explicitly."
                )
        else:
            self.stemmer = None

        # Disable expensive components we don't need for simple tokenization
        if disable is None:
            disable = ["parser", "ner"]

        try:
            self.nlp = spacy.load(model, disable=disable)
        except OSError as exc:
            raise ValueError(
                f"Spacy model '{model}' not found. "
                f"Install it using: python -m spacy download {model}"
            ) from exc

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text using spaCy with optional lemmatization or stemming.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens (lemmatized/stemmed if enabled)
        """
        doc = self.nlp(text.lower())

        # Filter out punctuation and spaces
        tokens = [
            token.text for token in doc if not token.is_punct and not token.is_space
        ]

        # Apply stemming if enabled
        if self.use_stemming and self.stemmer:
            return self.stemmer.stemWords(tokens)

        # Apply lemmatization if enabled (and stemming is not)
        if self.use_lemmatization:
            tokens = [
                token.lemma_
                for token in doc
                if not token.is_punct and not token.is_space
            ]

        return tokens

    def _detect_language_from_model(self, model: str) -> str | None:
        """Detect stemmer language from spaCy model name.

        Args:
            model: spaCy model name (e.g., "en_core_web_sm", "lt_core_news_sm")

        Returns:
            Language name for snowballstemmer, or None if not detected
        """
        # Extract language code from model name (first part before underscore)
        model_lower = model.lower()
        lang_code = model_lower.split("_")[0]

        return stemmer_language_map.get(lang_code)
