import spacy

from doxearch.tokenizer.tokenizer import Tokenizer


class SpacyTokenizer(Tokenizer):
    def __init__(
        self,
        model: str = "en_core_web_sm",
        use_lemmatization: bool = True,
        disable: list[str] | None = None,
    ):
        """Initialize SpacyTokenizer with a spacy language model.

        Args:
            model: Name of the spacy model to use (default: "en_core_web_sm")
             use_lemmatization: Whether to use lemmatization for word normalization
            disable: List of pipeline components to disable for faster processing.
                    Default disables parser and NER, keeping only tokenizer.
        """
        self.model_name = model
        self.use_lemmatization = use_lemmatization

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
        """Tokenize text using spaCy with optional lemmatization.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens (lemmatized if enabled)
        """
        doc = self.nlp(text.lower())

        if self.use_lemmatization:
            # Use lemma form to normalize different word forms
            tokens = [
                token.lemma_
                for token in doc
                if not token.is_punct and not token.is_space
            ]
        else:
            tokens = [
                token.text for token in doc if not token.is_punct and not token.is_space
            ]

        return tokens
