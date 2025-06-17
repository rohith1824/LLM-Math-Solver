import pytest
from src.tokenizer import Tokenizer

@pytest.fixture
def tok():
    samples = ["What is 123 + 45?", "Solve 8*7"]
    return Tokenizer(samples)

def test_simple_encode_decode(tok):
    text = "2+2=4"
    ids = tok.encode(text)
    recon = tok.decode(ids)
    assert recon == text

def test_multi_digit_reversal(tok):
    ids = tok.encode("123")
    # skip <START> and <END>
    middle = ids[1:-1]
    # reversed digits: ['3','2','1']
    assert [tok.id2token[i] for i in middle] == ["3","2","1"]

def test_unknown(tok):
    # use a char not in samples, e.g. '@'
    ids = tok.encode("@")
    # should map to <UNK> between START/END
    assert ids == [tok.token2id["<START>"],
                   tok.token2id["<UNK>"],
                   tok.token2id["<END>"]]
