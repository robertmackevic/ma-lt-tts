def lowercase(text: str) -> str:
    return text.lower()


def collapse_whitespace(text: str) -> str:
    return text.replace(' ', '')


def clean_oov_symbols(text: str) -> str:
    return (
        text
        .replace('„', '"')
        .replace('“', '"')
        .replace("–", "-")
    )


def basic_cleaners(text: str) -> str:
    text = lowercase(text)
    text = collapse_whitespace(text)
    text = clean_oov_symbols(text)

    return text
