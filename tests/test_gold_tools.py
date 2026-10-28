import hashlib
import json
from collections import Counter

import numpy as np

from gigarag import patterns
from gigarag.gold import Evidence, GoldItem, holdout_size, load_gold, save_gold, stratified_split
from gigarag.llm import CachedChat, ChatResponse, chat_request, read_env_file
from gigarag.passages import (
    Passage,
    document_passages,
    html_blocks,
    pack,
    passages_fingerprint,
    sections,
    split_article,
    split_long,
)
from gigarag.retrieval import Bm25Index, dense_search, top_k, union_ids
from gigarag.text import contains_quote, visible_text

# ---------- patterns ----------


def test_numeric_change_and_phrase_finder():
    assert patterns.has_numeric_change("Урон увеличен с 75/90 до 80/100")
    assert patterns.has_numeric_change("перезарядка уменьшена с 3 на 2")
    assert not patterns.has_numeric_change("теперь можно применять на союзников")
    find = patterns.build_phrase_finder(["Axe", "Black King Bar", "Bar"])
    assert find("Купил black king bar и пошёл на axe") == {"Black King Bar", "Axe"}
    assert find("Axes") == set()
    assert patterns.mentions_word("Пудж vs AXE", "axe")


def test_contains_quote_ignores_case_and_spacing():
    assert contains_quote("Урон  увеличен\nс 75 до 80", "урон увеличен с 75")
    assert not contains_quote("Урон увеличен", "урон уменьшен")
    assert not contains_quote("что угодно", "  ")


# ---------- passages ----------

ARTICLE = """<div itemprop="articleBody"><p>Вступление</p><h4>Axe</h4><ul><li>Броня +1</li></ul><p>Ещё</p>
<h4>Lion</h4><p>Hex дальше</p></div>"""


def test_article_sections_and_packing():
    assert split_article(ARTICLE, max_chars=1000) == ["Вступление", "Axe Броня +1 Ещё", "Lion Hex дальше"]
    assert sections([(False, "a"), (True, "H"), (False, "b")]) == [["a"], ["H", "b"]]
    assert pack(["aaaa", "bbbb", "cc"], max_chars=8) == ["aaaa", "bbbb cc"]  # пробел-разделитель учитывается


def test_nested_containers_lose_no_text():
    html = """<div itemprop="articleBody"><p>Интро</p><div><div><h5>Axe</h5><ul><li>Броня +1</li></ul>
    Голый <b>текст</b> в div<li>пункт вне списка</li></div></div></div>"""
    assert html_blocks(html) == [(False, "Интро"), (True, "Axe"), (False, "Броня +1"),
                                 (False, "Голый текст в div"), (False, "пункт вне списка")]
    article = {"doc_id": "a", "source": "dota2ru_article", "text": html}
    assert " ".join(p.text for p in document_passages(article)) == visible_text(html)


def test_split_long_respects_limit():
    text = "Первое предложение. Второе подлиннее! " + "слово " * 30
    pieces = pack([text], max_chars=25)
    assert all(len(p) <= 25 for p in pieces)
    assert " ".join(pieces).split() == text.split()
    assert split_long("x" * 10, max_chars=4) == ["xxxx", "xxxx", "xx"]


def test_document_passages_ids():
    post = {"doc_id": "tg:x/1", "source": "tg_x", "text": "<b>Патч</b> вышел"}
    assert [(p.passage_id, p.text) for p in document_passages(post)] == [("tg:x/1#0", "Патч вышел")]
    article = {"doc_id": "a", "source": "dota2ru_article", "text": ARTICLE}
    assert [p.passage_id for p in document_passages(article)] == ["a#0", "a#1", "a#2"]


# ---------- retrieval ----------

def test_top_k_and_union():
    assert top_k(np.array([0.1, 0.9, 0.5]), 2) == [(1, 0.9), (2, 0.5)]
    assert union_ids([[(1, 1.0), (2, 0.5)], [(3, 1.0), (1, 0.4)]]) == [1, 3, 2]


def test_bm25_and_dense_search():
    index = Bm25Index.build(["пудж получил нерф", "турнир в Бали", "нерф лины и пуджа"])
    assert index.search("нерф пудж", 1)[0][0] == 0
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert dense_search(vectors, np.array([0.1, 0.9]), 1)[0][0] == 1


# ---------- gold ----------

def items(n_per_type: dict[str, int]) -> list[GoldItem]:
    return [GoldItem(f"{t}{i}", t, "q", "a") for t, n in n_per_type.items() for i in range(n)]


def test_test_quota_bounds():
    assert [holdout_size(n, 0.3) for n in (1, 2, 3, 10, 15)] == [0, 1, 1, 3, 4]


def test_stratified_split_keeps_every_type_in_both_parts():
    gold = items({"factual": 15, "composite": 15, "abstract": 10, "negative": 10})
    split = stratified_split(gold, 0.3, seed=42)
    assert split == stratified_split(gold, 0.3, seed=42)
    per_type = Counter((g.qtype, split[g.qid]) for g in gold)
    assert per_type[("factual", "test")] == 4 and per_type[("abstract", "test")] == 3
    assert all(per_type[(t, s)] > 0 for t in ("factual", "negative") for s in ("test", "validation"))


def test_gold_roundtrip_and_relevant_docs(tmp_path):
    item = GoldItem("q1", "composite", "q", "a", evidence=(Evidence("d1#0", "d1", 2, "x"), Evidence("d1#1", "d1", 2),
                                                           Evidence("d2#0", "d2", 1)), styles=("typo",))
    save_gold([item], tmp_path / "g.jsonl")
    assert load_gold(tmp_path / "g.jsonl") == [item]
    assert item.relevant_docs() == ["d1"]
    assert item.relevant_docs(min_grade=1) == ["d1", "d2"]


# ---------- llm ----------

def test_chat_request_key_and_payload():
    a = chat_request("m", "sys", "user", 0.0, seed=1, json_mode=True)
    assert a.key() == chat_request("m", "sys", "user", 0.0, seed=1, json_mode=True).key()
    assert a.key() != chat_request("m", "sys", "user", 0.7, seed=1, json_mode=True).key()
    assert a.payload()["response_format"] == {"type": "json_object"}
    assert a.payload()["messages"][1] == {"role": "user", "content": "user"}
    assert "thinking" not in a.payload()


def test_chat_request_thinking_mode():
    default = chat_request("m", "sys", "user", 0.0, seed=1)
    off = chat_request("m", "sys", "user", 0.0, seed=1, thinking=False)
    assert off.payload()["thinking"] == {"type": "disabled"}
    assert chat_request("m", "sys", "user", 0.0, thinking=True).payload()["thinking"] == {"type": "enabled"}
    assert off.key() != default.key()
    # ключ запроса без явного режима не изменился с появлением поля: старые кэши остаются валидными
    assert default.key() == hashlib.sha256(json.dumps(
        {"model": "m", "messages": [["system", "sys"], ["user", "user"]], "temperature": 0.0, "seed": 1,
         "json_mode": False}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class FakeClient:
    def __init__(self):
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        return ChatResponse(json.dumps({"ok": True}), 10, 2)


def test_cached_chat_calls_api_once(tmp_path):
    client = FakeClient()
    chat = CachedChat(client, tmp_path)
    request = chat_request("m", "s", "u", 0.0)
    first, second = chat.complete(request), chat.complete(request)
    assert client.calls == 1 and not first.cached and second.cached and second.content == first.content


def test_read_env_file(tmp_path):
    (tmp_path / ".env").write_text("# comment\nA=1\nB = two=2\n")
    assert read_env_file(tmp_path / ".env") == {"A": "1", "B": "two=2"}
    assert read_env_file(tmp_path / "missing") == {}


def test_passages_fingerprint_tracks_text_and_order():
    a, b = Passage("d#0", "d", "раз"), Passage("d#1", "d", "два")
    assert passages_fingerprint([a, b]) == passages_fingerprint([a, b])
    assert passages_fingerprint([a, b]) != passages_fingerprint([b, a])
    assert passages_fingerprint([a, b]) != passages_fingerprint([a, Passage("d#1", "d", "три")])
