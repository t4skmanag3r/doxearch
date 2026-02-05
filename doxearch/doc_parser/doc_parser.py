from abc import ABC
from pathlib import Path

class DocParser(ABC):
    def parse(self, file_path: Path) -> str:
        raise NotImplementedError("Subclasses must implement this method")
    
    