import numpy as np
import pandas as pd
import pytest

from gigarag import cleaning, enrich, metadata, metrics, patterns, search
from gigarag.chunking import Chunk, ChunkerSpec, child_chunks, chunk_boundaries, chunk_documents, chunk_length_stats
from gigarag.corpus import Document
from gigarag.dedup import connected_components
from gigarag.gold import Evidence, GoldItem
from gigarag.relevance import EvidenceChunks, build_qrels, lost_evidence, match_words, quote_coverage


def doc(doc_id: str, text: str, source: str = "tg_x", date: str = "2024-01-01", **meta) -> Document:
    return Document(doc_id, source, date, None, "", "", text, meta)


# ---------- patterns ----------

def test_tags_to_line_breaks_and_tidy_lines():
    text = patterns.tags_to_line_breaks("a<br>b<BR/>c</p><p>d</div>")
    assert patterns.tidy_lines(patterns.strip_tags(text, " ")) == "a\nb\nc\nd"
    assert patterns.tidy_lines("  a   b \n\n\n\n c\t d \n") == "a b\n\nc d"


def test_decorative_symbols_keep_numbers_and_punctuation():
    cleaned = patterns.tidy_lines(patterns.remove_decorative_symbols("• Урон +10% (1/2/3) | #новости ✔ «ок»"))
    assert cleaned == "Урон +10% (1/2/3) новости «ок»"


def test_remove_invisible_chars():
    assert patterns.remove_invisible_chars("пере­нос​ и ❤️") == "перенос и ❤"


# ---------- cleaning ----------

def test_html_to_text_keeps_lines():
    assert cleaning.html_to_text(doc("d", "<b>Патч</b> 7.36<br>Урон&nbsp;+5</div>")).split() == \
        ["Патч", "7.36", "Урон", "+5"]
    assert "\n" in cleaning.html_to_text(doc("d", "a<br>b"))


def test_normalize_text_nfkc_and_spaces():
    assert cleaning.normalize_text("ﬁ a  b\n\n\n\nc") == "fi a b\n\nc"


def test_channel_signatures_strip_only_repeated_last_lines():
    docs = [doc(f"tg:{i}", f"пост {i}\n@Dota2") for i in range(3)] + [doc("tg:9", "@Dota2\nконец")]
    signatures = cleaning.channel_signatures(docs, min_repeats=3)
    assert signatures == {"tg_x": frozenset({"@Dota2"})}
    assert cleaning.strip_trailing_lines("пост\n@Dota2\n@Dota2", signatures["tg_x"]) == "пост"
    assert cleaning.strip_trailing_lines("@Dota2\nконец", signatures["tg_x"]) == "@Dota2\nконец"


def test_exact_duplicates_keep_earliest_and_drop_empty():
    corpus = cleaning.CleanCorpus((doc("b", "Тот же  текст", date="2024-02-01"), doc("a", "тот же текст"),
                                   doc("e", "  ")))
    result = cleaning.drop_exact_duplicates(corpus)
    assert [d.doc_id for d in result.docs] == ["a"]
    assert result.aliases == {"b": "a"}


def test_canonical_doc_id_follows_chain():
    assert cleaning.canonical_doc_id({"a": "b", "b": "c"}, "a") == "c"
    assert cleaning.canonical_doc_id({}, "a") == "a"


def test_near_duplicates_keep_longest():
    base = " ".join(f"слово{i}" for i in range(40))
    corpus = cleaning.CleanCorpus((doc("short", base), doc("long", base + " ещё"), doc("other", "совсем другой текст")))
    result = cleaning.drop_near_duplicates(corpus)
    assert {d.doc_id for d in result.docs} == {"long", "other"}
    assert result.aliases == {"short": "long"}


def test_run_stages_names_and_order():
    corpus = cleaning.CleanCorpus((doc("a", "<b>Привет</b> https://x.io 🔥"),))
    states = cleaning.run_stages(corpus, cleaning.PIPELINE[:4])
    assert [name for name, _ in states] == ["html", "unicode", "urls", "emoji"]
    assert states[-1][1].docs[0].text == "Привет"


def test_connected_components():
    assert connected_components([("b", "c"), ("a", "b"), ("x", "y")]) == [["a", "b", "c"], ["x", "y"]]


# ---------- relevance и метрики ----------

def test_quote_coverage_and_match_words():
    assert match_words("<b>Ёжик</b>  ИДЁТ") == ["ежик", "идет"]
    assert quote_coverage(["a", "b", "c", "d"], ["b", "c", "x"]) == pytest.approx(2 / 3)
    assert quote_coverage(["a"], []) == 0.0


def test_build_qrels_uses_alias_and_split_quote():
    chunks = [Chunk("keep@0", "keep", 0, "один два три"), Chunk("keep@1", "keep", 1, "четыре пять"),
              Chunk("other@0", "other", 0, "один два три четыре пять")]
    item = GoldItem("q1", "factual", "?", "!", (Evidence("removed#0", "removed", 2, "два три четыре"),))
    qrels = build_qrels([item], chunks, aliases={"removed": "keep"})
    assert qrels["q1"][0].chunks == frozenset({0})  # бо́льшая часть цитаты; чужой документ не считается
    assert lost_evidence(build_qrels([item], chunks)) == [("q1", "removed#0")]


def test_retrieval_metrics():
    evidence = [EvidenceChunks("p1", 2, frozenset({3})), EvidenceChunks("p2", 2, frozenset({7, 8})),
                EvidenceChunks("p3", 1, frozenset({1}))]
    ranking = [1, 3, 5, 8]
    assert metrics.hit_at_k(ranking, evidence, 1) == 0.0
    assert metrics.hit_at_k(ranking, evidence, 2) == 1.0
    assert metrics.recall_at_k(ranking, evidence, 2) == 0.5
    assert metrics.recall_at_k(ranking, evidence, 4) == 1.0
    assert metrics.precision_at_k(ranking, evidence, 4) == 0.5
    assert metrics.reciprocal_rank(ranking, evidence) == 0.5


def test_paired_changes():
    per_q = pd.DataFrame({"split": "validation", "retriever": "bm25", "qid": ["a", "b", "a", "b"],
                          "config": ["x", "x", "y", "y"], "R@5": [0.5, 1.0, 1.0, 1.0]})
    assert metrics.paired_changes(per_q, "x", "y").loc["bm25"].to_dict() == {"лучше": 1, "хуже": 0,
                                                                              "без изменений": 1}


# ---------- поиск ----------

def test_chunk_scores_take_max_over_entries_and_respect_mask():
    entry_scores = np.array([0.1, 0.9, 0.5, 0.3])
    owners = np.array([0, 0, 1, 2])
    assert search.chunk_scores(entry_scores, owners, 4).tolist() == [0.9, 0.5, 0.3, -np.inf]
    assert search.rank_chunks(entry_scores, owners, 4) == [0, 1, 2]
    assert search.rank_chunks(entry_scores, owners, 4, allowed=np.array([False, True, True, True])) == [1, 2]


# ---------- чанкинг ----------

def test_chunk_documents_and_children():
    docs = [doc("a", "x y z"), doc("b", "")]
    chunks = chunk_documents(docs, lambda texts: [t.split() for t in texts])
    assert [c.chunk_id for c in chunks] == ["a@0", "a@1", "a@2"]
    children, owners = child_chunks(chunks[:2], lambda texts: [[t, t] for t in texts])
    assert [c.chunk_id for c in children] == ["a@0/0", "a@0/1", "a@1/0", "a@1/1"] and owners == [0, 0, 1, 1]


def test_chunk_boundaries_and_stats():
    assert chunk_boundaries(["ab", "cde", "f"]) == frozenset({2, 5})
    stats = chunk_length_stats([10, 60, 600])
    assert stats["<50_%"] == pytest.approx(100 / 3) and stats[">500_%"] == pytest.approx(100 / 3)


def test_chunker_spec_label():
    assert ChunkerSpec("token").label == "token(512)"
    assert ChunkerSpec("semantic", threshold=0.5, skip_window=1, embedder="BAAI/bge-m3").label == \
        "sdpm(512, t=0.5, skip=1, bge-m3)"


# ---------- метаданные ----------

def test_patch_at_and_language_and_type():
    releases = [("2024-01-01", "7.35"), ("2024-02-01", "7.35b")]
    assert metadata.patch_at("2024-01-15", releases) == "7.35"
    assert metadata.patch_at("2024-02-01T10:00", releases) == "7.35b"
    assert metadata.patch_at("2023-12-31", releases) is None
    assert metadata.language("Урон увеличен") == "ru" and metadata.language("Black King Bar") == "en"
    assert metadata.content_type(doc("t", "", topical=False)) == metadata.OFFTOPIC
    assert metadata.content_type(doc("p", "", source="patchnotes")) == "патчноут"


def test_patch_filter_and_prefix():
    meta = pd.DataFrame({"patch": ["7.35", "7.36", None]})
    allowed = metadata.patch_filter(meta, ["", "", "обсуждаем 7.35"])
    assert allowed(meta, "что в 7.35?").tolist() == [True, False, True]
    assert allowed(meta, "без патча") is None
    row = {"title": "Заголовок", "content_type": "пост", "patch": "7.35", "date": "2024-01-15T10:00"}
    assert metadata.with_metadata_prefix("текст", row) == "Заголовок | пост | патч 7.35 | 2024-01-15\nтекст"


# ---------- LLM-обработка ----------

def test_enrich_helpers():
    assert enrich.parse_questions('{"questions": [" a ", "", 3, "b"]}') == ["a", "b"]
    assert enrich.eligible([5, 30, 100], min_tokens=30) == [1, 2]
    assert enrich.Usage(1, 1_000_000, 1_000_000, 0).cost_usd == pytest.approx(sum(enrich.PRICE_PER_MILLION))


def test_complete_all_sends_duplicate_requests_once():
    from gigarag.llm import ChatResponse

    class CountingChat:
        def __init__(self):
            self.calls = []

        def complete(self, request):
            self.calls.append(request.key())
            return ChatResponse(request.messages[-1][1].upper(), 1, 1)

    chat = CountingChat()
    requests = [enrich.summary_request(t) for t in ["а", "б", "а", "а"]]
    assert [r.content for r in enrich.complete_all(chat, requests)] == ["А", "Б", "А", "А"]
    assert len(chat.calls) == 2
