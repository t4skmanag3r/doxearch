from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Tokenize a single text.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
