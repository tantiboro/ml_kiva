import pytest
from ml_kiva.preprocess import load_and_clean_data
from ml_kiva.preprocess import clean_text


def test_html_removal():
    input_text = "Loan for <b>running</b> a farm.<br/>"
    expected = "loan run farm" # 'run' is the lemma of 'running'
    assert clean_text(input_text) == expected


def test_non_ascii_removal():
    input_text = "Supporting families in Bogotá 😊"
    # Now that we added accent normalization, this will pass
    assert "bogota" in clean_text(input_text)


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
