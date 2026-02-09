from doxearch.tokenizer.spacy_tokenizer.spacy_tokenizer import SpacyTokenizer


def main():
    # Initialize the tokenizer with Lithuanian model
    print("Initializing SpacyTokenizer with Lithuanian model...")
    try:
        tokenizer = SpacyTokenizer(model="lt_core_news_sm")
        print("✓ Lithuanian tokenizer initialized successfully")
    except ValueError as e:
        print(f"✗ Error initializing tokenizer: {e}")
        print("\nTo install the Lithuanian model, run:")
        print("  uv run python -m spacy download lt_core_news_sm")
        return

    # Test texts in Lithuanian
    test_texts = [
        "Labas rytas! Kaip laikaisi?",
        "Lietuva yra graži šalis su turtinga istorija.",
        "Vilnius, Kaunas ir Klaipėda yra didžiausi Lietuvos miestai.",
        "Šiandien oras labai gražus, saulėta ir šilta.",
    ]

    print("\n=== Testing Lithuanian Tokenization ===\n")

    for i, text in enumerate(test_texts, 1):
        print(f"Text {i}: {text}")
        tokens = tokenizer.tokenize(text)
        print(f"Tokens ({len(tokens)}): {tokens}")
        print()

    # Test with English text for comparison
    print("=== Testing with English Text ===\n")
    english_text = "Hello world! This is a test of the tokenizer."
    print(f"Text: {english_text}")
    tokens = tokenizer.tokenize(english_text)
    print(f"Tokens ({len(tokens)}): {tokens}")
    print()

    # Test edge cases
    print("=== Testing Edge Cases ===\n")

    # Empty string
    empty_tokens = tokenizer.tokenize("")
    print(f"Empty string tokens: {empty_tokens} (length: {len(empty_tokens)})")

    # Only punctuation
    punct_tokens = tokenizer.tokenize(".,!?;:")
    print(f"Punctuation only tokens: {punct_tokens} (length: {len(punct_tokens)})")

    # Mixed case
    mixed_text = "LABAS Rytas kaip LAIKAISI?"
    mixed_tokens = tokenizer.tokenize(mixed_text)
    print(f"Mixed case text: {mixed_text}")
    print(f"Tokens: {mixed_tokens}")

    print("\n✓ All tokenization tests completed successfully!")


if __name__ == "__main__":
    main()
