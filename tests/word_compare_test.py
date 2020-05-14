#!/usr/bin/env python3

import pytest
from ..sentence_compare import sentence_compare

address_true = {
    "speech": ["Speak to people in a formal way."],
    "corpus": [
        "the particulars of the place where someone lives or an organization is situated.",
        "a formal speech delivered to an audience.",
        "write the name and address of the intended recipient on (an envelope, letter, or package).",
        "speak to (a person or an assembly), typically in a formal way.",
    ],
}

address_false = {
    "speech": ["Where I walk to."],
    "corpus": [
        "the particulars of the place where someone lives or an organization is situated.",
        "a formal speech delivered to an audience.",
        "write the name and address of the intended recipient on (an envelope, letter, or package).",
        "speak to (a person or an assembly), typically in a formal way.",
    ],
}

package_false = {
    "speech": ["No Idea."],
    "corpus": [
        "an object or group of objects wrapped in paper or packed in a box.",
        "a set of proposals or terms offered or agreed as a whole.",
        "put into a box or wrapping for sale or transport.",
        "present (someone or something) in an attractive or advantageous way.",
    ],
}

package_true = {
    "speech": ["An item wrapped in paper."],
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
        (address_true, "a formal speech delivered to an audience."),
        (address_false, False),
        (package_false, False),
        (
            package_true,
            "an object or group of objects wrapped in paper or packed in a box.",
        ),
    ],
)
def test_sentence_compare(string_input, expected):
    assert sentence_compare(string_input["speech"], string_input["corpus"]) == expected
