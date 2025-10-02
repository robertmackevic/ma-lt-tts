import re

from src.text.symbols import ACCENTS


def remove_stress_marks(text: str) -> str:
    return re.sub(rf"[{re.escape(ACCENTS)}]", "", text)


def collapse_whitespace(text: str) -> str:
    return text.replace(" ", "")


def normalize_text(text: str) -> str:
    return (
        re.sub(r"\s+", " ", re.sub(r"[‐‑–—―]", "-", text))
        .replace("…", "...")
        .replace("\ufeff", "")
        .replace("'", "")
        .replace("̇", "")
        .replace("„", "\"")
        .replace("“", "\"")
        .strip()
        .lower()
    )


def filter_punctuations(text: str) -> str:
    return normalize_text("".join(
        character
        if character.isalpha() or character.isspace()
        else " " for character in text
    ))


def basic_cleaners(text: str) -> str:
    text = normalize_text(text)
    text = collapse_whitespace(text)
    return text
