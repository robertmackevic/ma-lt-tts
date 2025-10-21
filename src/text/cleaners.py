import re

from src.text.symbols import ACCENTS


def remove_stress_marks(text: str) -> str:
    return re.sub(rf"[{re.escape(ACCENTS)}]", "", text)


def normalize_text(text: str, lower: bool = True) -> str:
    text = (
        re.sub(r"\s+", " ", re.sub(r"[‐‑–—―]", "-", text))
        .replace("…", "...")
        .replace("\ufeff", "")
        .replace("'", "")
        .replace("̇", "")
        .replace("„", "\"")
        .replace("“", "\"")
        .strip()
    )

    return text.lower() if lower else text


def filter_punctuations(text: str) -> str:
    return normalize_text("".join(
        character
        if character.isalpha() or character.isspace() or character in ACCENTS
        else " " for character in text
    ))


def basic_cleaners(text: str) -> str:
    text = normalize_text(text)
    return text
