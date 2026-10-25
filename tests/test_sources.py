import random

import pytest

from gigarag.corpus import Document, duplicate_ids, ensure_unique_ids
from gigarag.http import RetryPolicy, is_retryable
from gigarag.sources import dota2ru, patch_at_least, telegram
from gigarag.sources.patchnotes import Names, patch_documents, render_hero, render_note, unique_suffixes
from gigarag.sources.topic_lexicon import build_topic_matcher

NAMES = Names(heroes={1: "Anti-Mage"}, abilities={5003: "Mana Break", 1: "Blink Dagger"})


# ---------- patchnotes ----------

def test_render_note_indent_and_hidden_dot():
    assert render_note({"indent_level": 2, "note": "x"}) == "  - x"
    assert render_note({"hide_dot": True, "note": "<br>"}) == "<br>"


def test_render_hero_order():
    hero = {"hero_id": 1, "hero_notes": [{"note": "броня +1"}],
            "abilities": [{"ability_id": 5003, "ability_notes": [{"note": "урон +5"}]}],
            "subsections": [{"title": "Аспект", "general_notes": [{"note": "новый"}]}],
            "talent_notes": [{"note": "талант 10"}]}
    assert render_hero(hero, NAMES) == ["- броня +1", "Mana Break:", "- урон +5", "[Аспект]", "- новый",
                                        "Таланты:", "- талант 10"]


def test_unique_suffixes():
    assert unique_suffixes([("i", "a"), ("i", "b"), ("i", "a"), ("i", "a")]) == ["", "", "#2", "#3"]


def test_patch_documents_ids_and_general_items():
    data = {"items": [{"ability_id": -1, "ability_notes": [{"note": "a"}]},
                      {"ability_id": -1, "ability_notes": [{"note": "b"}]},
                      {"ability_id": 1, "ability_notes": [{"note": "c"}]}],
            "heroes": [{"hero_id": 1, "hero_notes": [{"note": "d"}]}]}
    docs = patch_documents("7.38", 1722409200, data, NAMES)
    assert [d.doc_id for d in docs] == ["patch:7.38:items:Предметы: общее", "patch:7.38:items:Предметы: общее#2",
                                        "patch:7.38:items:Blink Dagger", "patch:7.38:hero:Anti-Mage"]
    assert docs[2].text == "Патч 7.38 — Blink Dagger\n- c"
    assert docs[0].date == "2024-07-31"


def test_patch_at_least():
    assert patch_at_least("7.34", "7.34") and patch_at_least("7.41f", "7.34")
    assert not patch_at_least("7.33e", "7.34")
    assert not patch_at_least(None, "7.34")


# ---------- dota2.ru ----------

ARTICLE = """<h1>В Dota 2 вышел патч 7.41f</h1><time datetime="2026-09-15T21:49:07+03:00"></time>
<div itemprop="articleBody"><p>Список изменений</p></div>
<div class="global-content__comment-wrap">
  <a class="global-content__comment-name">Daimon_1</a><span data-time="1789498532"></span>
  <a data-reply-answer="1748898">ответить</a>
  <div class="global-content__comment-user--message">мда <blockquote>цитата</blockquote></div>
</div>
<div class="global-content__comment-wrap">
  <div class="global-content__comment-user--message">без id</div>
</div>
<div class="global-content__comment-wrap"><p>без сообщения</p></div>"""


def test_parse_article_and_comments():
    ref, article = dota2ru.parse_article(ARTICLE, "67852-patc-7-41f", "7.41f", fallback_date="2000-01-01")
    assert ref.article_id == "67852" and ref.date == "2026-09-15" and ref.title.startswith("В Dota 2")
    assert article.doc_id == "dota2ru:article:67852" and "Список изменений" in article.text
    comments = dota2ru.parse_comments(ARTICLE, ref)
    assert len(comments) == 2
    first, second = comments
    assert first.doc_id == "dota2ru:comment:1748898"
    assert first.meta == {"article_id": "67852", "author": "Daimon_1", "has_quote": True}
    assert second.doc_id == "dota2ru:comment:" + dota2ru.comment_fallback_id("67852", second.text)
    assert second.date == ref.date


def test_comment_fallback_id_is_stable():
    assert dota2ru.comment_fallback_id("1", "<p>x</p>") == dota2ru.comment_fallback_id("1", "<p>x</p>")
    assert dota2ru.comment_fallback_id("1", "<p>x</p>") != dota2ru.comment_fallback_id("2", "<p>x</p>")


# ---------- Telegram ----------

def post(pid: int, date: str, text: str, channel: str = "Dota2") -> str:
    return (f'<div class="tgme_widget_message" data-post="{channel}/{pid}">'
            f'<div class="tgme_widget_message_text">{text}</div><time datetime="{date}T10:00:00+00:00"></time></div>')


def test_parse_and_merge_pages_keeps_first_seen():
    page1 = telegram.parse_feed_page(post(2, "2024-01-02", "v1") + post(3, "2024-01-03", "c"), "Dota2")
    page2 = telegram.parse_feed_page(post(1, "2024-01-01", "a") + post(2, "2024-01-02", "v2"), "Dota2")
    merged = telegram.merge_pages([page1, page2])
    assert [d.doc_id for d in merged] == ["tg:Dota2/1", "tg:Dota2/2", "tg:Dota2/3"]
    assert merged[1].text == '<div class="tgme_widget_message_text">v1</div>'  # сырой HTML, как в источнике
    assert merged[0].meta["has_photo"] is False and merged[0].url == "https://t.me/Dota2/1"


def test_in_window_and_quota():
    doc = Document("id", "tg_x", "2024-01-01", None, "u", "t", "text")
    assert telegram.in_window(doc, "2023-08-08")
    assert not telegram.in_window(doc, "2024-02-01")
    assert telegram.offtopic_quota(topical_count=10, share=0.2, available=1) == 1
    assert telegram.offtopic_quota(topical_count=10, share=0.2, available=50) == 2


def test_select_with_offtopic_is_deterministic_and_marks_topical():
    posts = [Document(f"tg:c/{i}", "tg_c", f"2024-01-{i:02d}", None, "u", "t", "патч" if i <= 5 else "мем")
             for i in range(1, 21)]
    is_topical = build_topic_matcher([])
    a = telegram.select_with_offtopic(posts, is_topical, 0.4, random.Random("42:c"))
    b = telegram.select_with_offtopic(posts, is_topical, 0.4, random.Random("42:c"))
    assert a == b
    assert sum(d.meta["topical"] for d in a) == 5 and len(a) == 7
    assert [d.date for d in a] == sorted(d.date for d in a)
    assert "topical" not in posts[0].meta  # исходные документы не изменены


# ---------- corpus и http ----------

def test_document_json_key_order_and_unique_ids():
    doc = Document("a", "s", None, None, "u", "t", "x", {"k": 1})
    assert doc.to_json().startswith('{"doc_id": "a", "source": "s", "date": null, "patch": null')
    assert duplicate_ids([doc, doc]) == ["a"]
    with pytest.raises(ValueError):
        ensure_unique_ids([doc, doc])


def test_retry_policy():
    assert [RetryPolicy(base_delay=5).backoff(i) for i in range(3)] == [5, 10, 20]
    assert is_retryable(429) and is_retryable(503) and not is_retryable(404)
