from __future__ import annotations

import pytest

from dutchbook.elicit import SYSTEM_PROMPT, build_prompt, parse


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("Some thoughts.\n<probability>0.35</probability>", 0.35),
        ("<probability>0</probability>", 0.0),
        ("<probability>1</probability>", 1.0),
        ("<probability>1.00</probability>", 1.0),
        ("<probability> .5 </probability>", 0.5),
        ("<probability>0.125</probability>\n", 0.125),
    ],
)
def test_accepts_plain_decimals(text: str, value: float) -> None:
    parsed = parse(text)
    assert parsed.ok
    assert parsed.probability == value


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("I think about 40%.", "no_tag"),
        ("<probability>40%</probability>", "bad_number"),
        ("<probability>1.2</probability>", "bad_number"),
        ("<probability>-0.1</probability>", "bad_number"),
        ("<probability>about 0.4</probability>", "bad_number"),
        ("<probability>0.3-0.4</probability>", "bad_number"),
        ("<probability>0.3</probability> or <probability>0.4</probability>", "multiple_tags"),
        ("<Probability>0.3</Probability>", "no_tag"),
        ("I can't predict the outcome of a future election.", "refusal"),
        ("I'm unable to forecast this.", "refusal"),
    ],
)
def test_rejects_everything_else(text: str, status: str) -> None:
    parsed = parse(text)
    assert parsed.status == status
    assert parsed.probability is None


def test_prompt_carries_the_question_and_format() -> None:
    prompt = build_prompt("Will it rain in Paris on 1 June 2027?")
    assert prompt.startswith("Question: Will it rain in Paris on 1 June 2027?")
    assert "<probability>0.XX</probability>" in prompt
    assert "3 November 2026" in SYSTEM_PROMPT
