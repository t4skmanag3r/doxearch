import spacy

from doxearch.tokenizer.tokenizer import Tokenizer


class SpacyTokenizer(Tokenizer):
    def __init__(self, model: str = "en_core_web_sm"):
        """Initialize SpacyTokenizer with a spacy language model.

        Args:
            model: Name of the spacy model to use (default: "en_core_web_sm")
        """
        try:
            self.nlp = spacy.load(model)
        except OSError as exc:
            raise ValueError(
                f"Spacy model '{model}' not found. "
                f"Install it using: python -m spacy download {model}"
            ) from exc

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text using spacy.

        Args:
            text: Input text to tokenize

        Returns:
            List of lowercase tokens, excluding punctuation and whitespace
        """
        doc = self.nlp(text)
        tokens = [
            token.text.lower()
            for token in doc
            if not token.is_punct and not token.is_space
        ]
        return tokens
