from abc import ABC

class Tokenizer(ABC):
    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError("Subclasses must implement this method")