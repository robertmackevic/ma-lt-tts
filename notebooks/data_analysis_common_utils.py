import os
import random
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from phonemizer import phonemize
from phonemizer.separator import Separator

from src.model import commons
from src.text.cleaners import filter_punctuations, remove_stress_marks
from src.text.convert import text_to_sequence


def preprocess_text_for_synthesis(
        text: str,
        text_cleaners: list[str],
        language: str,
        phonemized: bool,
        stressed: bool
) -> torch.LongTensor:
    text_norm = text_to_sequence(text, text_cleaners, language, phonemized, stressed)
    text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def phonemize_series(
        series: pd.Series,
        language: str = "lt",
        backend: str = "espeak",
        njobs: int | None = None,
        phone_sep: str = "",
        word_sep: str = " ",
        preserve_punctuation: bool = True
) -> pd.Series:
    """
    Batch-phonemizes a Pandas Series of texts using phonemizer.

    - language="lt" (Lithuanian)
    - phone_sep controls the separator between phonemes
    - word_sep controls the separator between words
    """
    if njobs is None:
        njobs = max(1, os.cpu_count() or 1)

    texts = series.fillna("").astype(str).tolist()
    phonemes = phonemize(
        texts,
        language=language,
        backend=backend,
        strip=True,
        njobs=njobs,
        preserve_punctuation=preserve_punctuation,
        separator=Separator(phone=phone_sep, syllable="", word=word_sep),
    )
    return pd.Series(phonemes, index=series.index)


def phonemize_text(
        text: str,
        language: str = "lt",
        backend: str = "espeak",
        phone_sep: str = "",
        word_sep: str = " ",
        preserve_punctuation: bool = True
) -> str:
    phonemes = phonemize(
        [text],
        language=language,
        backend=backend,
        strip=True,
        njobs=1,
        preserve_punctuation=preserve_punctuation,
        separator=Separator(phone=phone_sep, syllable="", word=word_sep),
    )[0]

    return phonemes


def _filter_text(text: str) -> str:
    return filter_punctuations(remove_stress_marks(text))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_filelist_as_df(filelist_path: str | Path) -> pd.DataFrame:
    filelist_path = Path(filelist_path)
    _df = pd.read_csv(filelist_path, sep="|", header=None, names=["filepath", "sentence"], dtype=str)
    _df["filepath"] = _df["filepath"].str.strip()
    _df["sentence"] = _df["sentence"].str.strip()
    return _df


def summarize_duration(_df: pd.DataFrame) -> None:
    durations = _df["duration[s]"]

    desc = durations.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])

    plt.figure(figsize=(8, 5))
    plt.hist(durations, bins=50, color="lightgray", edgecolor="black")
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")
    plt.title("Distribution of Clip Durations")
    plt.show()

    total_seconds = durations.sum()
    total_hours = total_seconds / 3600

    print("Summary statistics for duration[s]:")
    print(desc)
    print(f"Total duration: {total_seconds:.2f} seconds ({total_hours:.2f} hours)")


def summarize_sentences(_df: pd.DataFrame) -> None:
    words = [word for text in _df["sentence"] for word in filter_punctuations(text).lower().split()]
    word_counts_per_text = [len(_filter_text(text).split()) for text in _df["sentence"]]
    character_counts_per_text = [len(text) for text in _df["sentence"]]
    character_counts_per_word = [len(word) for word in words]

    kwargs = {
        "edgecolor": "black",
        "color": "gray",
        "alpha": 0.5,
        "linewidth": 1.2
    }

    plt.figure(figsize=(6, 6))
    plt.hist(word_counts_per_text, bins=15, **kwargs)
    plt.xlabel("Word Count per Sample")
    plt.ylabel("Frequency")
    plt.title("Distribution of Word Counts")
    plt.grid(True, linestyle="--", color="gray")
    plt.tight_layout()
    plt.show()
    print(f"""
    Word statistics (Per Text):
        Total:  {sum(word_counts_per_text):>10}
        Unique: {len(set(words)):>10}
        Mean:   {mean(word_counts_per_text) if len(word_counts_per_text) > 0 else 0:>10.2f}
        STD:    {stdev(word_counts_per_text) if len(word_counts_per_text) > 1 else 0:>10.2f}
        Median: {median(word_counts_per_text) if len(word_counts_per_text) > 0 else 0:>10.2f}
        Min:    {min(word_counts_per_text) if len(word_counts_per_text) > 0 else 0:>10}
        Max:    {max(word_counts_per_text) if len(word_counts_per_text) > 0 else 0:>10}
    """)

    plt.figure(figsize=(6, 6))
    plt.hist(character_counts_per_text, bins=40, **kwargs)
    plt.xlabel("Character Count per Sample")
    plt.ylabel("Frequency")
    plt.title("Distribution of Character Counts (per Sequence)")
    plt.grid(True, linestyle="--", color="gray")
    plt.tight_layout()
    plt.show()
    print(f"""
    Character statistics (Per Text):
        Total:  {sum(character_counts_per_text):>10}
        Mean:   {mean(character_counts_per_text) if len(character_counts_per_text) > 0 else 0:>10.2f}
        STD:    {stdev(character_counts_per_text) if len(character_counts_per_text) > 1 else 0:>10.2f}
        Median: {median(character_counts_per_text) if len(character_counts_per_text) > 0 else 0:>10.2f}
        Min:    {min(character_counts_per_text) if len(character_counts_per_text) > 0 else 0:>10}
        Max:    {max(character_counts_per_text) if len(character_counts_per_text) > 0 else 0:>10}
    """)

    plt.figure(figsize=(6, 6))
    plt.hist(character_counts_per_word, bins=15, **kwargs)
    plt.xlabel("Character Count per Word")
    plt.ylabel("Frequency")
    plt.title("Distribution of Character Counts (per Word)")
    plt.grid(True, linestyle="--", color="gray")
    plt.tight_layout()
    plt.show()
    print(f"""
    Character statistics (Per Word):
        Total:  {sum(character_counts_per_word):>10}
        Mean:   {mean(character_counts_per_word) if len(character_counts_per_word) > 0 else 0:>10.2f}
        STD:    {stdev(character_counts_per_word) if len(character_counts_per_word) > 1 else 0:>10.2f}
        Median: {median(character_counts_per_word) if len(character_counts_per_word) > 0 else 0:>10.2f}
        Min:    {min(character_counts_per_word) if len(character_counts_per_word) > 0 else 0:>10}
        Max:    {max(character_counts_per_word) if len(character_counts_per_word) > 0 else 0:>10}
    """)
