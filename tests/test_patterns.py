from gigarag import patterns


def test_strip_tags_replaces_each_tag():
    assert patterns.strip_tags("<b>Pudge</b> сильный", " ") == " Pudge  сильный"
    assert patterns.strip_tags("<span>Общие</span>", "") == "Общие"


def test_entity_and_url_spans():
    text = 'a&nbsp;b <a href="https://t.me/Dota2">x</a>'
    assert patterns.entity_spans(text) == [(1, 7)]
    assert patterns.has_html_entity(text)
    assert [text[s:e] for s, e in patterns.url_spans(text)] == ["https://t.me/Dota2"]


def test_character_class_spans():
    text = "Урон 75, x_y!"
    assert [text[s:e] for s, e in patterns.letter_spans(text)] == ["Урон", "x", "y"]
    assert [text[s:e] for s, e in patterns.digit_spans(text)] == ["75"]
    assert [text[s:e] for s, e in patterns.punctuation_spans(text)] == [",", "!"]
    assert [text[s:e] for s, e in patterns.special_spans(text)] == [",", "_", "!"]


def test_collapse_whitespace_and_tokens():
    assert patterns.collapse_whitespace("  a \n\t b  ") == "a b"
    assert patterns.word_tokens("бкб, 7.41f!") == ["бкб", "7", "41f"]
    assert patterns.topic_words("ап нерф Blink 2") == ["нерф", "Blink"]


def test_patch_versions():
    assert patterns.parse_patch_version("7.36c") == (36, "c")
    assert patterns.parse_patch_version("6.88") is None
    assert patterns.patch_from_slug("67852-patc-7-41f-v-dota-2") == "7.41f"
    assert patterns.patch_from_slug("69116-cto-mortalles-zdet") is None


def test_page_links_and_post_ids():
    html = '<a href="/news/1-a">x</a><a href="/news/2-b">y</a><a href="/news/1-a">z</a><a href="/news/3-c?page=2">'
    assert patterns.news_links(html) == ["/news/1-a", "/news/2-b"]  # ссылки с параметрами — не статьи
    assert patterns.page_numbers('href="/news/1-a?page=2" href="/news/1-a?page=3"', "/news/1-a") == {2, 3}
    feed = '<div data-post="Dota2/15"></div><div data-post="dota2/16"></div><time datetime="2024-01-02T10:00:00+00:00">'
    assert patterns.telegram_post_ids(feed, "Dota2") == [15, 16]
    assert patterns.datetime_attrs(feed) == ["2024-01-02T10:00:00+00:00"]


def test_ads_marker_ignores_game_bonus():
    assert patterns.has_ads_marker("Забирай фрибет по промокоду")
    assert not patterns.has_ads_marker("бонус к урону увеличен")


def test_any_matcher_uses_word_boundaries():
    match = patterns.build_any_matcher([patterns.word_boundary_pattern("пак"), r"нерф"])
    assert match("ПАК опять сильный")
    assert match("понерфили")
    assert not match("упаковка")
