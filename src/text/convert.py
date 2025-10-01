from typing import Iterable, List

from src.text import cleaners
from src.text.symbols import get_vocabulary


def text_to_sequence(
        text: str, cleaner_names: Iterable[str], language: str, phonemized: bool, stressed: bool
) -> List[int]:
    text = clean_text(text, cleaner_names)

    return cleaned_text_to_sequence(text, language, phonemized, stressed)


def cleaned_text_to_sequence(text: str, language: str, phonemized: bool, stressed: bool) -> List[int]:
    _, symbol_to_id, _ = get_vocabulary(language, phonemized, stressed)

    return [symbol_to_id[symbol] for symbol in text]


def clean_text(text: str, cleaner_names: Iterable[str]) -> str:
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        text = cleaner(text)

    return text
