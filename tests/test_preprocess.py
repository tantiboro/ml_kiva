import pytest
from src.preprocess import clean_text


def test_html_removal():
    input_text = "Loan for <b>farming</b> equipment.<br/>"
    expected = "loan farm equipment"
    # Note: 'farm' is the lemma of 'farming'
    assert clean_text(input_text) == expected


def test_non_ascii_removal():
    input_text = "Supporting families in Bogotá 😊"
    # Should remove the accent and the emoji
    assert "bogota" in clean_text(input_text)
    assert "😊" not in clean_text(input_text)


def test_lemmatization():
    input_text = "The borrower was running three businesses"
    # 'running' -> 'run', 'businesses' -> 'business', 'was' is a stopword
    result = clean_text(input_text)
    assert "run" in result
    assert "business" in result
    assert "running" not in result


def test_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""
