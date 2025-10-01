from functools import cache
from typing import List, Mapping, Tuple

ACCENTS = u'\u0300\u0301\u0303'
IPA = 'ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘̩ᵻ̈͡'


@cache
def get_vocabulary(
        language: str, phonemized: bool, stressed: bool
) -> Tuple[List[str], Mapping[str, int], Mapping[int, str]]:
    if language != "lt":
        raise NotImplementedError(f"Unsupported language: {language}")

    symbols = (
            ["_"]  # padding
            + list('";:,.!?—-\'"() ')
            + list("abcdefghijklmnopqrstuvwxyząčęėįšųūž")
            + (list(ACCENTS) if stressed else [])
            + (list(IPA) if phonemized else [])
    )

    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    id_to_symbol = dict(enumerate(symbols))

    return symbols, symbol_to_id, id_to_symbol
