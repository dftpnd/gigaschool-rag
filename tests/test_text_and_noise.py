import numpy as np
import pandas as pd

from gigarag import noise
from gigarag.noise import TokenClass
from gigarag.text import is_mostly_latin, latin_letter_share, plain_lower, visible_text


def test_visible_text_decodes_entities_after_stripping_tags():
    assert visible_text("<p>Урон&nbsp;увеличен</p>\n<br>с&#160;75 &quot;до&quot; 80") == 'Урон увеличен с 75 "до" 80'


def test_plain_lower():
    assert plain_lower("<b>ПАТЧ</b>&amp;Нерф") == " патч &нерф"


def test_latin_share():
    assert latin_letter_share("abcж") == 0.75
    assert latin_letter_share("123") == 0.0
    assert is_mostly_latin("Black King Bar")
    assert not is_mostly_latin("Пудж и бкб")


def test_char_classes_priority():
    text = '<a href="https://x.io">7 ок!</a>🔥'
    codes = noise.char_classes(text)
    by_char = {text[i]: TokenClass(codes[i]) for i in range(len(text))}
    assert TokenClass(codes[0]) == TokenClass.HTML                       # тег
    assert TokenClass(codes[text.index("https")]) == TokenClass.URL       # URL внутри тега важнее тега
    assert by_char["7"] == TokenClass.NUM
    assert by_char["о"] == TokenClass.WORD
    assert by_char["!"] == TokenClass.PUNCT
    assert by_char["🔥"] == TokenClass.EMOJI


def test_token_class_counts_uses_first_char_of_token():
    text = "ab 12"

    def offsets(_):
        return [(0, 2), (2, 2), (3, 5)]  # пустой токен пропускается

    counts = noise.token_class_counts(text, offsets)
    assert counts[TokenClass.WORD] == 1 and counts[TokenClass.NUM] == 1 and counts.sum() == 2


def test_class_shares_and_removable_noise():
    counts = pd.DataFrame({"source": ["a", "b"], "word": [3, 1], "num": [1, 0], "punct": [0, 0], "special": [0, 0],
                           "emoji": [0, 1], "html": [0, 2], "url": [0, 0]})
    shares = noise.class_shares_percent(counts, by="source")
    assert np.allclose(shares.sum(axis=1), 100)
    assert shares.loc["b", "html"] == 50
    assert noise.removable_noise_percent(shares).loc["ВСЕГО"] == 3 / 8 * 100
