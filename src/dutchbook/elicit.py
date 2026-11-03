"""The prompt every proposition is asked with, and the strict parser for the answer.

Every question goes out in a fresh conversation with the same system prompt and the
same instructions, so the only thing that changes between two calls is the question
text. The model may reason first, but the reply has to end with exactly one
`<probability>` tag holding a plain decimal between 0 and 1. Anything else is counted
as a parse failure or a refusal, never repaired or guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

AS_OF = "3 November 2026"

SYSTEM_PROMPT = (
    "You are a forecaster. Each message asks for the probability of one event that has "
    "not been decided yet. You have no tools and cannot look anything up, so answer from "
    f"what you already know. Today's date is {AS_OF}."
)

INSTRUCTIONS = (
    "Give your probability that the answer is yes. You may reason briefly first. End "
    "your reply with your final answer on its own line, in exactly this format:\n"
    "<probability>0.XX</probability>\n"
    "The number must be a decimal between 0 and 1. If you are unsure, give your best "
    "estimate anyway."
)


def build_prompt(question: str) -> str:
    return f"Question: {question}\n\n{INSTRUCTIONS}"


TAG = re.compile(r"<probability>(.*?)</probability>", re.DOTALL)
NUMBER = re.compile(r"\s*(0(?:\.\d+)?|1(?:\.0+)?|0?\.\d+)\s*")
# Phrases that mark a reply with no tag as a refusal rather than a formatting slip.
REFUSAL = re.compile(
    r"\b(I can(?:no|')t|I can not|I'm (?:not able|unable)|I am (?:not able|unable)|"
    r"unable to (?:provide|give|predict|forecast)|I won't|I will not|I must decline|"
    r"I (?:have to|need to) decline|not (?:able|possible) to (?:predict|forecast|know))\b",
    re.IGNORECASE,
)

STATUSES = ("ok", "refusal", "no_tag", "multiple_tags", "bad_number", "error")


@dataclass(frozen=True)
class Parsed:
    status: str  # one of STATUSES
    probability: float | None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def parse(text: str) -> Parsed:
    tags = TAG.findall(text)
    if not tags:
        if REFUSAL.search(text):
            return Parsed("refusal", None, _snippet(text))
        return Parsed("no_tag", None, _snippet(text))
    if len(tags) > 1:
        return Parsed("multiple_tags", None, " | ".join(t.strip() for t in tags)[:120])
    m = NUMBER.fullmatch(tags[0])
    if m is None:
        return Parsed("bad_number", None, tags[0].strip()[:120])
    return Parsed("ok", float(m.group(1)))


def _snippet(text: str) -> str:
    return " ".join(text.split())[:160]
