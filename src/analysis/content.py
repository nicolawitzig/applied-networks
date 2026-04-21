"""
Content analysis (RQ4): topic + tonality of tweets mentioning the official
accounts.

The paper used a trained student coder and Krippendorff's alpha on a 100-tweet
sample. These rule-based classifiers are a weak baseline so the pipeline runs
end-to-end; swap in a supervised model (or manual codes) once you have a
training set in data/coded/.
"""
from __future__ import annotations

import re

TOPIC_LEXICON = {
    "academic": [
        "research", "study", "publication", "paper", "journal",
        "teaching", "lecture", "seminar", "course", "thesis",
        "conference", "workshop", "symposium",
        "forschung", "lehre", "vorlesung", "studie",
    ],
    "organizational": [
        "ranking", "rector", "president", "alumni", "career",
        "hiring", "job", "position", "vacancy",
        "sustainability", "finance", "budget",
        "covid", "pandemic", "measure",
        "stelle", "karriere",
    ],
}

TONALITY_LEXICON = {
    "positive": ["congrat", "proud", "great", "excellent", "thank", "award", "wonderful", "glückwunsch", "stolz", "danke"],
    "negative": ["disappoint", "fail", "criticism", "problem", "shame", "unfair", "terrible", "enttäusch", "kritik", "skandal"],
}


def _count_matches(text: str, terms: list[str]) -> int:
    t = text.lower()
    return sum(1 for term in terms if term in t)


def classify_topic(text: str) -> str:
    scores = {topic: _count_matches(text, terms) for topic, terms in TOPIC_LEXICON.items()}
    if max(scores.values(), default=0) == 0:
        return "other"
    return max(scores, key=scores.get)


def classify_tonality(text: str) -> str:
    pos = _count_matches(text, TONALITY_LEXICON["positive"])
    neg = _count_matches(text, TONALITY_LEXICON["negative"])
    if pos == 0 and neg == 0:
        return "neutral"
    return "positive" if pos >= neg else "negative"


MENTION_PATTERN = re.compile(r"@(\w+)")


def mentions_any(text: str, handles: list[str]) -> bool:
    found = {m.lower() for m in MENTION_PATTERN.findall(text)}
    return any(h.lower() in found for h in handles)
