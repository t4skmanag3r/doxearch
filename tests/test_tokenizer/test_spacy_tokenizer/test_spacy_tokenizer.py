import pytest

from doxearch.tokenizer.spacy_tokenizer.spacy_tokenizer import SpacyTokenizer


@pytest.fixture
def spacy_tokenizer():
    """Fixture to create a SpacyTokenizer instance."""
    return SpacyTokenizer()


def test_spacy_tokenizer_basic(spacy_tokenizer):
    """Test basic tokenization."""
    text = "Hello world! This is a test."
    tokens = spacy_tokenizer.tokenize(text)

    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert "hello" in tokens
    assert "world" in tokens
    assert "test" in tokens


def test_spacy_tokenizer_lowercase(spacy_tokenizer):
    """Test that tokens are converted to lowercase."""
    text = "HELLO World TeSt"
    tokens = spacy_tokenizer.tokenize(text)

    assert all(token.islower() for token in tokens)
    assert "hello" in tokens
    assert "world" in tokens
    assert "test" in tokens


def test_spacy_tokenizer_removes_punctuation(spacy_tokenizer):
    """Test that punctuation is removed."""
    text = "Hello, world! How are you?"
    tokens = spacy_tokenizer.tokenize(text)

    assert "," not in tokens
    assert "!" not in tokens
    assert "?" not in tokens
    assert "hello" in tokens
    assert "world" in tokens


def test_spacy_tokenizer_removes_whitespace(spacy_tokenizer):
    """Test that whitespace tokens are removed."""
    text = "Hello    world\n\nTest"
    tokens = spacy_tokenizer.tokenize(text)

    assert "\n" not in tokens
    assert "  " not in tokens
    assert " " not in tokens
    assert len(tokens) == 3


def test_spacy_tokenizer_empty_string(spacy_tokenizer):
    """Test tokenization of empty string."""
    text = ""
    tokens = spacy_tokenizer.tokenize(text)

    assert isinstance(tokens, list)
    assert len(tokens) == 0


def test_spacy_tokenizer_only_punctuation(spacy_tokenizer):
    """Test tokenization of string with only common punctuation."""
    text = ".,!?;:"
    tokens = spacy_tokenizer.tokenize(text)

    assert isinstance(tokens, list)
    # Common punctuation should be filtered out
    assert len(tokens) == 0


def test_spacy_tokenizer_complex_text(spacy_tokenizer):
    """Test tokenization of more complex text."""
    text = "The quick brown fox jumps over the lazy dog."
    tokens = spacy_tokenizer.tokenize(text)

    expected_tokens = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "the",
        "lazy",
        "dog",
    ]
    assert tokens == expected_tokens


def test_spacy_tokenizer_with_numbers(spacy_tokenizer):
    """Test tokenization with numbers."""
    text = "I have 3 apples and 5 oranges."
    tokens = spacy_tokenizer.tokenize(text)

    assert "3" in tokens
    assert "5" in tokens
    assert "apples" in tokens
    assert "oranges" in tokens


def test_spacy_tokenizer_invalid_model():
    """Test that invalid model raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        SpacyTokenizer(model="invalid_model_name")

    assert "not found" in str(exc_info.value)
    assert "python -m spacy download" in str(exc_info.value)


def test_spacy_tokenizer_custom_model():
    """Test initialization with custom model (if available)."""
    try:
        tokenizer = SpacyTokenizer(model="en_core_web_sm")
        assert tokenizer.nlp is not None
    except ValueError:
        pytest.skip("en_core_web_sm model not installed")
