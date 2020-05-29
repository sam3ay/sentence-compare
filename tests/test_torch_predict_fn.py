#!/usr/bin/env python3

import pytest
from model.code.torch_constructor import predict_fn
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "/workspaces/sentence-compare/model/roberta-large-nli-stsb-mean-tokens"
)

address_true = {
    "query": ["Speak to people in a formal way."],
    "corpus": [
        "the particulars of the place where someone lives or an organization is situated.",
        "a formal speech delivered to an audience.",
        "write the name and address of the intended recipient on (an envelope, letter, or package).",
        "speak to (a person or an assembly), typically in a formal way.",
    ],
}

address_false = {
    "query": ["Where I walk to."],
    "corpus": [
        "the particulars of the place where someone lives or an organization is situated.",
        "a formal speech delivered to an audience.",
        "write the name and address of the intended recipient on (an envelope, letter, or package).",
        "speak to (a person or an assembly), typically in a formal way.",
    ],
}

package_false = {
    "query": ["No Idea."],
    "corpus": [
        "an object or group of objects wrapped in paper or packed in a box.",
        "a set of proposals or terms offered or agreed as a whole.",
        "put into a box or wrapping for sale or transport.",
        "present (someone or something) in an attractive or advantageous way.",
    ],
}

package_true = {
    "query": ["An item wrapped in paper."],
    "corpus": [
        "a set of proposals or terms offered or agreed as a whole.",
        "put into a box or wrapping for sale or transport.",
        "an object or group of objects wrapped in paper or packed in a box.",
        "present (someone or something) in an attractive or advantageous way.",
    ],
}


@pytest.mark.parametrize(
    ("string_input", "expected"),
    [
        (
            address_true,
            {"predict": 1, "sentence": "a formal speech delivered to an audience."},
        ),
        (
            address_false,
            {
                "predict": 0,
                "sentence": "write the name and address of the intended recipient on (an envelope, letter, or package).",
            },
        ),
        (
            package_false,
            {
                "predict": 0,
                "sentence": "a set of proposals or terms offered or agreed as a whole.",
            },
        ),
        (
            package_true,
            {
                "predict": 1,
                "sentence": "an object or group of objects wrapped in paper or packed in a box.",
            },
        ),
    ],
)
def test_sentence_compare(string_input, expected):
    assert predict_fn(string_input, model) == expected
