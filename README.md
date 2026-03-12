# Doxearch

Local search engine for indexing documents (PDFs, docxs, etc.) using TF-IDF.

## Overview

Doxearch is a powerful **local document search engine** that uses **TF-IDF** (Term Frequency-Inverse Document Frequency) algorithm to index and search through your documents. It supports multiple document formats and provides both command-line and graphical user interfaces.

## Features

- **Multiple Document Formats**: Support for PDF and DOCX files
- **TF-IDF Search Algorithm**: Fast and relevant search results using industry-standard ranking
- **Fuzzy Matching**: Find documents even with typos or variations in search terms
- **Multi-Language Support**: Built-in support for not just English through spaCy models with easy installation
- **Smart Indexing**: Incremental indexing with hash-based change detection
- **Directory Management**: Manage multiple document directories with different configurations

## Requirements

 **None if using standalone binary executable**

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) environment manager
- spaCy with language models (en_core_web_sm and others)

## Installation

### Binary Executable

Download the latest pre-built executable from the [Releases](https://github.com/t4skmanag3r/doxearch/releases) page:

- **Windows**: `doxearch-gui.exe`
- **Linux**: `doxearch-gui`

No installation required - just download and run!

### PyPI Package

[WiP]

### From Source

1. Clone the repository:
```bash
git clone https://github.com/t4skmanag3r/doxearch.git
cd doxearch
```

2. Install uv (if not already installed):
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

3. Install dependencies:
```bash
uv sync --all-extras
```

### Command-Line Interface (CLI)

[WiP]

### Graphical User Interface (GUI)

Launch the GUI either through the **binary executable** or by running `uv run doxearch-gui` from the command line.

**The GUI provides:**
- Search Tab: Interactive search with real-time results
- Index Tab: Configure and index document directories
- Documents Tab: Browse indexed documents
- Directories Tab: Manage multiple document directories
- Models Tab: Download and manage spaCy language models

## Building Standalone Executable

Build a standalone executable using Nuitka:

```bash
uv run python build_nuitka.py
```

The executable will be created in the current directory:
- Windows: doxearch-gui.exe
- Linux: doxearch-gui

## Project Structure

```
doxearch/
├── doxearch/              # Core library
│   ├── doc_index/         # Document indexing system
│   ├── doc_parser/        # Document parsers (PDF, DOCX)
│   ├── tokenizer/         # Text tokenization
│   └── tf_idf/            # TF-IDF implementation
├── doxearch_cli/          # Command-line interface
├── doxearch_gui/          # Graphical user interface
└── tests/                 # Test suite
```

## How It Works
1. Indexing: Documents are parsed, tokenized, and indexed using TF-IDF scoring
2. Tokenization: Text is processed using spaCy with optional lemmatization/stemming
3. Storage: Document metadata and term frequencies are stored in SQLite databases
4. Search: Queries are tokenized and matched against indexed documents with relevance scoring
5. Fuzzy Matching: Similar terms are found using Levenshtein distance for typo tolerance

## Configuration
Doxearch stores configuration and indexes in:
- Linux/macOS: ~/.local/share/doxearch/
- Windows: %APPDATA%\doxearch\
Each indexed directory has its own SQLite database with:
- Document metadata (path, hash, filename, term count, last indexed time)
- Term frequencies for TF-IDF calculation
- Tokenization settings

## Development

### Running Tests

```bash
uv run pytest
```

### Development Setup

1. Install development dependencies:
```bash
uv sync --all-extras
```

### Code Quality
The project uses:
- black for code formatting
- mypy for type checking
- isort for import sorting
- automated checks using pytest

## License
This project is licensed under the MIT License - see [LICENSE](./LICENSE) for details

## Contributing
Contributions are welcome, just submit a pull request, thank you!

## Support
For issues, questions, or contributions, please open an issue on GitHub.