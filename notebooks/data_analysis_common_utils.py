import os
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib.pyplot as plt
import pandas as pd
from phonemizer import phonemize
from phonemizer.separator import Separator

from src.text.cleaners import filter_punctuations, remove_stress_marks


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
    - backend="espeak" (or "espeak-mbrola")
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
        # Choose how results are joined:
        separator=Separator(phone=phone_sep, syllable="", word=word_sep),
    )
    return pd.Series(phonemes, index=series.index)


def _filter_text(text: str) -> str:
    return filter_punctuations(remove_stress_marks(text))


def load_filelist_as_df(filelist_path: str | Path) -> pd.DataFrame:
    filelist_path = Path(filelist_path)
    _df = pd.read_csv(filelist_path, sep="|", header=None, names=["filepath", "sentence"], dtype=str)
    _df["filepath"] = _df["filepath"].str.strip()
    _df["sentence"] = _df["sentence"].str.strip()
    return _df


def summarize_duration(_df: pd.DataFrame) -> None:
    desc = _df["duration[s]"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    plt.figure(figsize=(8, 5))
    plt.hist(_df["duration[s]"], bins=50, color="lightgray", edgecolor="black")
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")
    plt.title("Distribution of Clip Durations")
    plt.show()
    print("Summary statistics for duration[s]:")
    print(desc)


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
