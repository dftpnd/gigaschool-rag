"""Построение золотого набора КТ 1 по стадиям; каждая стадия читает результат предыдущей.

    python build_gold.py generate   # затравки → кандидаты (LLM)
    python build_gold.py stylize    # лексическое разнообразие части вопросов (LLM)
    python build_gold.py label      # кандидаты отрывков (BM25 + e5 + BGE-M3) → LLM-судья 0/1/2
    python build_gold.py review     # автоматические флаги и лист для ручного ревью
    python build_gold.py finalize   # решения ревью → gold/gold.jsonl с разбиением validation/test

Ответы LLM кэшируются (data/llm_cache): повторный запуск воспроизводит результат и ничего не стоит.
"""

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, replace

import gold_prompts as P
import numpy as np
from gold_seeds import Seed, all_seeds

import gigarag.embeddings
import gigarag.passages
import gigarag.patterns
import gigarag.text
from gigarag.cache import CacheStore, code_fingerprint, load_or_compute, read_npy, write_npy
from gigarag.corpus import read_jsonl
from gigarag.dedup import jaccard, word_shingles
from gigarag.embeddings import encode_passages
from gigarag.gold import RELEVANT, Candidate, Evidence, GoldItem, review_digest, save_gold, stratified_split
from gigarag.llm import CachedChat, ChatRequest, ChatResponse, OpenAIChat, chat_request, parse_json_object
from gigarag.parallel import parallel_map
from gigarag.passages import Passage, corpus_passages, passages_fingerprint
from gigarag.patterns import patch_mentions, word_tokens
from gigarag.retrieval import Bm25Index, dense_search, union_ids
from gigarag.text import contains_quote, unsupported_words, visible_text
from settings import CACHE, CORPUS, DATA, ENV_FILE, GOLD_DIR, LLM_CACHE, SEED

WORK = DATA / "gold"
REVIEW_FILE = GOLD_DIR / "review.json"

QUOTAS = {"factual": 15, "composite": 15, "abstract": 10, "negative": 10}
TEST_SHARE = 0.3
SEED_DOC_CHARS = 1500             # обрезка документа-затравки в промпте генератора
PER_RETRIEVER_K = 10              # кандидатов отрывков на (поисковик × запрос)
MAX_JUDGED = 40                   # потолок отрывков на вопрос для судьи (плюс отрывки затравок)
DUPLICATE_QUESTION_JACCARD = 0.6  # вопросы с таким сходством биграмм слов считаются дублями
MIN_EVIDENCE_CHARS = 40           # отрывок короче не может сам по себе подтверждать ответ
# Стили на 20 кандидатов: опечатки ~15%, остальные стили по 10%, ~45% без изменений.
STYLE_CYCLE = ["typo"] * 3 + ["colloquial", "name_variant", "short", "long"] * 2 + [""] * 9
PASSAGE_EMB_CODE = code_fingerprint(gigarag.patterns, gigarag.text, gigarag.passages, gigarag.embeddings)
EMBEDDERS = {  # модель → (ревизия, префикс запроса, префикс отрывка, размер батча)
    "intfloat/multilingual-e5-small": ("614241f622f53c4eeff9890bdc4f31cfecc418b3", "query: ", "passage: ", 128),
    # BGE-M3 крупнее: батч поменьше, чтобы уместиться в свободную видеопамять рядом с другими процессами.
    "BAAI/bge-m3": ("5617a9f61b028005a4858fdac845db406aefb181", "", "", 16),
}


# ---------- общее ----------

def load_corpus() -> tuple[dict[str, dict], str]:
    rows, sha = read_jsonl(CORPUS)
    return {r["doc_id"]: {**r, "visible": visible_text(r["text"])} for r in rows}, sha


def llm() -> CachedChat:
    return CachedChat(OpenAIChat.from_env(ENV_FILE), LLM_CACHE)


def write_stage(name: str, records: list[dict]) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                                        encoding="utf-8")


def read_stage(name: str) -> list[dict]:
    return [json.loads(line) for line in (WORK / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]


def candidate_from(record: dict) -> Candidate:
    return Candidate(**{**record, "seed_docs": tuple(record["seed_docs"]), "seed_quotes": tuple(record["seed_quotes"]),
                        "styles": tuple(record["styles"])})


# ---------- generate ----------

def generation_request(seed: Seed, index: int, docs: dict[str, dict]):
    if seed.qtype == "negative":
        rng = random.Random(f"{SEED}:negative:{index}")
        examples = [docs[d]["visible"][:200] for d in rng.sample(sorted(docs), 3)]
        user = P.negative_user(seed.kind, examples)
        return chat_request(P.GENERATOR_MODEL, P.NEGATIVE_SYSTEM, user, P.GENERATION_TEMPERATURE, P.LLM_SEED, True)
    documents = [(d, docs[d]["visible"][:SEED_DOC_CHARS]) for d in seed.doc_ids]
    user = P.generator_user(seed.qtype, documents)
    return chat_request(P.GENERATOR_MODEL, P.GENERATOR_SYSTEM, user, P.GENERATION_TEMPERATURE, P.LLM_SEED, True)


def to_candidate(seed: Seed, index: int, content: dict, docs: dict[str, dict]) -> Candidate:
    evidence = content.get("evidence", [])
    quotes_ok = all(e.get("doc_id") in docs and contains_quote(docs[e["doc_id"]]["visible"], e.get("quote", ""))
                    for e in evidence)
    return Candidate(
        cid=f"c{index:03d}", qtype=seed.qtype, question=content["question"].strip(), answer=content["answer"].strip(),
        seed_docs=seed.doc_ids, seed_quotes=tuple(e.get("quote", "") for e in evidence),
        original_question=content["question"].strip(),
        meta={"kind": seed.kind, "quotes_ok": quotes_ok, "why_absent": content.get("why_absent", ""),
              "evidence_docs": [e.get("doc_id") for e in evidence]},
    )


def stage_generate() -> None:
    docs, _ = load_corpus()
    seeds = all_seeds(list(docs.values()), SEED)
    chat = llm()
    responses = parallel_map(lambda pair: chat.complete(generation_request(pair[1], pair[0], docs)),
                             list(enumerate(seeds)))
    candidates = [to_candidate(s, i, parse_json_object(r.content), docs)
                  for i, (s, r) in enumerate(zip(seeds, responses, strict=True))]
    write_stage("candidates", [asdict(c) for c in candidates])
    report_usage("generate", responses)
    print(Counter(c.qtype for c in candidates))


# ---------- stylize ----------

def assign_styles(candidates: list[Candidate]) -> dict[str, str]:
    """cid → стиль по циклу STYLE_CYCLE в перемешанном порядке; `name_variant` — только при латинице в вопросе."""
    order = sorted(c.cid for c in candidates)
    random.Random(f"{SEED}:styles").shuffle(order)
    by_id = {c.cid: c for c in candidates}

    def fitting(style: str, c: Candidate) -> str:
        has_latin = any(ch.isascii() and ch.isalpha() for ch in c.question)
        return "colloquial" if style == "name_variant" and not has_latin else style
    return {cid: fitting(STYLE_CYCLE[i % len(STYLE_CYCLE)], by_id[cid]) for i, cid in enumerate(order)}


def stage_stylize() -> None:
    candidates = [candidate_from(r) for r in read_stage("candidates")]
    styles = assign_styles(candidates)
    targets = [c for c in candidates if styles[c.cid]]
    chat = llm()
    requests = [chat_request(P.GENERATOR_MODEL, P.STYLE_SYSTEM, P.style_user(styles[c.cid], c.question),
                             P.STYLE_TEMPERATURE, P.LLM_SEED, True) for c in targets]
    responses = parallel_map(chat.complete, requests)
    rewritten = {c.cid: parse_json_object(r.content)["question"].strip()
                 for c, r in zip(targets, responses, strict=True)}
    styled = [replace(c, question=rewritten.get(c.cid, c.question), styles=(styles[c.cid],) if styles[c.cid] else ())
              for c in candidates]
    write_stage("styled", [asdict(c) for c in styled])
    report_usage("stylize", responses)
    print(Counter(styles.values()))


# ---------- label ----------

def passage_embeddings(passages: list[Passage]) -> dict[str, np.ndarray]:
    """Эмбеддинги отрывков по моделям. Кэш привязан к отпечатку упорядоченных отрывков и коду
    нарезки и кодирования; размер загруженной матрицы сверяется с числом отрывков."""
    texts = [p.text for p in passages]
    cache = CacheStore(CACHE, passages_fingerprint(passages))

    def embed(model: str, rev: str, prefix: str, batch: int) -> np.ndarray:
        vectors = load_or_compute(
            cache.path("passage_emb", "npy", model=model, revision=rev, prefix=prefix, max_seq_length=512,
                       code=PASSAGE_EMB_CODE),
            lambda: encode_passages(texts, model, rev, max_seq_length=512, prefix=prefix, batch_size=batch),
            load=read_npy, save=write_npy)
        if vectors.shape[0] != len(passages):
            raise ValueError(f"{model}: в кэше {vectors.shape[0]} векторов, отрывков {len(passages)}")
        return vectors
    return {model: embed(model, rev, p_prefix, batch) for model, (rev, _, p_prefix, batch) in EMBEDDERS.items()}


def retrieval_queries(c: Candidate) -> list[str]:
    """Запросы для поиска кандидатов: вопрос, исходная формулировка и (кроме негативных) эталонный ответ."""
    queries = [c.question, c.original_question] + ([c.answer] if c.qtype != "negative" else [])
    return list(dict.fromkeys(q for q in queries if q))


def candidate_passages(candidates: list[Candidate], passages: list[Passage]) -> dict[str, list[int]]:
    """cid → индексы отрывков для судьи: объединение выдач трёх поисковиков плюс отрывки документов-затравок."""
    bm25 = Bm25Index.build([p.text for p in passages])
    vectors = passage_embeddings(passages)
    queries = {c.cid: retrieval_queries(c) for c in candidates}
    flat = sorted({q for qs in queries.values() for q in qs})
    query_vectors = {model: dict(zip(flat, encode_passages(flat, model, rev, max_seq_length=512, prefix=q_prefix,
                                                           batch_size=batch), strict=True))
                     for model, (rev, q_prefix, _, batch) in EMBEDDERS.items()}
    by_doc: dict[str, list[int]] = {}
    for i, p in enumerate(passages):
        by_doc.setdefault(p.doc_id, []).append(i)

    def for_candidate(c: Candidate) -> list[int]:
        rankings = [bm25.search(q, PER_RETRIEVER_K) for q in queries[c.cid]]
        rankings += [dense_search(vectors[m], query_vectors[m][q], PER_RETRIEVER_K)
                     for m in EMBEDDERS for q in queries[c.cid]]
        seeds = [i for d in c.seed_docs for i in by_doc.get(d, [])]
        return list(dict.fromkeys(union_ids(rankings)[:MAX_JUDGED] + seeds))
    return {c.cid: for_candidate(c) for c in candidates}


def judge_request(c: Candidate, passage: Passage) -> ChatRequest:
    return chat_request(P.JUDGE_MODEL, P.JUDGE_SYSTEM, P.judge_user(c.question, c.answer, passage.text),
                        P.JUDGE_TEMPERATURE, P.LLM_SEED, True)


def quote_retry_request(first: ChatRequest, reply: str) -> ChatRequest:
    """Тот же диалог с судьёй плюс просьба скопировать цитату дословно."""
    return replace(first, messages=(*first.messages, ("assistant", reply), ("user", P.QUOTE_RETRY_USER)))


def judge_passage(chat: CachedChat, c: Candidate, passage: Passage) -> tuple[dict, list[ChatResponse]]:
    """Оценка отрывка с проверенной цитатой. Положительная оценка без дословной цитаты — повторный запрос;
    если и он не дал цитату, оценка обнуляется, а исходная сохраняется в `unverified_grade` для ревью."""
    request = judge_request(c, passage)
    first = chat.complete(request)
    verdict = parse_json_object(first.content)
    responses = [first]
    if int(verdict["grade"]) > 0 and not contains_quote(passage.text, verdict.get("quote", "")):
        retry = chat.complete(quote_retry_request(request, first.content))
        verdict, responses = parse_json_object(retry.content), [first, retry]
    grade, quote = int(verdict["grade"]), verdict.get("quote", "")
    verified = grade == 0 or contains_quote(passage.text, quote)
    label = {"passage_id": passage.passage_id, "doc_id": passage.doc_id,
             "grade": grade if verified else 0, "quote": quote if verified and grade > 0 else ""}
    return (label if verified else {**label, "unverified_grade": grade}), responses


def stage_label() -> None:
    docs, _ = load_corpus()
    candidates = [candidate_from(r) for r in read_stage("styled")]
    passages = list(corpus_passages(docs.values()))
    pools = candidate_passages(candidates, passages)
    jobs = [(c, passages[i]) for c in candidates for i in pools[c.cid]]
    print("пар на разметку:", len(jobs))
    chat = llm()
    results = parallel_map(lambda job: judge_passage(chat, *job), jobs, workers=12)
    labels: dict[str, list[dict]] = {c.cid: [] for c in candidates}
    for (c, _), (label, _) in zip(jobs, results, strict=True):
        labels[c.cid].append(label)
    write_stage("labeled", [{**asdict(c), "labels": labels[c.cid]} for c in candidates])
    report_usage("label", [r for _, responses in results for r in responses])
    print("повторов цитаты:", sum(len(responses) > 1 for _, responses in results),
          "снято положительных без цитаты:", sum("unverified_grade" in label for label, _ in results))


# ---------- review ----------

def question_bigrams(text: str) -> frozenset[str]:
    return word_shingles(word_tokens(text.lower()), n=2)


def other_patch_evidence(record: dict, doc_patch: dict[str, str | None]) -> list[str]:
    """Отрывки с оценкой 2 из документов других патчей, чем названы в вопросе: судья часто засчитывает
    правку той же сущности из соседнего патча. Для вопроса без номеров патчей — пусто."""
    asked = set(patch_mentions(record["original_question"])) | set(patch_mentions(record["question"]))
    return [x["passage_id"] for x in record["labels"] if x["grade"] >= RELEVANT and asked
            and doc_patch.get(x["doc_id"]) and doc_patch[x["doc_id"]] not in asked]


def auto_flags(record: dict, all_records: list[dict], doc_patch: dict[str, str | None]) -> list[str]:
    """Причины для отклонения или правки разметки, которые проверяются автоматически."""
    relevant = [x for x in record["labels"] if x["grade"] >= RELEVANT]
    flags = []
    other_patch = other_patch_evidence(record, doc_patch)
    if other_patch:
        flags.append(f"оценка 2 у отрывков других патчей: {', '.join(other_patch)}")
    if record["qtype"] == "negative" and relevant:
        flags.append(f"ответ найден в корпусе ({len(relevant)} отр.)")
    if record["qtype"] != "negative":
        if not {x["doc_id"] for x in relevant} & set(record["seed_docs"]):
            flags.append("судья не подтвердил документ-затравку")
        if not record["meta"]["quotes_ok"]:
            flags.append("цитата генератора не найдена в документе")
    unverified = [x for x in record["labels"] if "unverified_grade" in x]
    if unverified:
        flags.append(f"судья не дал дословную цитату — оценка снята у {len(unverified)} отр.")
    mine = question_bigrams(record["question"])
    duplicates = [o["cid"] for o in all_records if o["cid"] < record["cid"]
                  and jaccard(mine, question_bigrams(o["question"])) >= DUPLICATE_QUESTION_JACCARD]
    if duplicates:
        flags.append(f"дубль вопроса {duplicates[0]}")
    return flags


def stage_review() -> None:
    docs, _ = load_corpus()
    passage_text = {p.passage_id: p.text for p in corpus_passages(docs.values())}
    doc_patch = {doc_id: d["patch"] for doc_id, d in docs.items()}
    records = read_stage("labeled")
    lines = []
    for r in records:
        flags = auto_flags(r, records, doc_patch)
        relevant = [x for x in r["labels"] if x["grade"] >= RELEVANT]
        unsupported = unsupported_words(r["answer"], [passage_text[x["passage_id"]] for x in relevant])
        lines += [f"## {r['cid']} · {r['qtype']} · {r['meta']['kind']} · стиль: {','.join(r['styles']) or '—'}"
                  f" · digest {review_digest(r)}",
                  f"Q: {r['question']}", *([f"Q0: {r['original_question']}"] if r["styles"] else []),
                  f"A: {r['answer']}", f"флаги: {'; '.join(flags) or 'нет'}",
                  f"слова ответа вне доказательств: {', '.join(unsupported) or 'нет'}",
                  f"релевантных отрывков: {len(relevant)} из {len(r['labels'])}; затравки: {', '.join(r['seed_docs'])}",
                  *[f"  - [{x['passage_id']}] {x['quote'][:160]}" for x in relevant], ""]
    (WORK / "review_sheet.md").write_text("\n".join(lines), encoding="utf-8")
    print("лист ревью:", WORK / "review_sheet.md")


# ---------- finalize ----------

def apply_overrides(labels: list[dict], overrides: list[dict]) -> list[dict]:
    """Правки ревьюера поверх разметки судьи: оценка заменяется или добавляется; цитата — из правки,
    а если её там нет, остаётся цитата судьи."""
    by_passage = {x["passage_id"]: x for x in labels}
    for o in overrides:
        judged = by_passage.get(o["passage_id"], {})
        by_passage[o["passage_id"]] = {"passage_id": o["passage_id"], "doc_id": o["passage_id"].rsplit("#", 1)[0],
                                       "grade": o["grade"], "quote": o.get("quote", judged.get("quote", ""))}
    return list(by_passage.values())


def is_substantive(passage_text: str) -> bool:
    """Отрывок достаточно длинный, чтобы сам по себе служить доказательством (а не «100%» или «+1»)."""
    return len(passage_text) >= MIN_EVIDENCE_CHARS


def to_gold_item(record: dict, decision: dict, qid: str, passage_text: dict[str, str]) -> GoldItem:
    labels = apply_overrides(record["labels"], decision.get("evidence_overrides", []))
    kept = [x for x in labels if x["grade"] >= 1 and is_substantive(passage_text[x["passage_id"]])]
    evidence = tuple(Evidence(x["passage_id"], x["doc_id"], x["grade"], x["quote"])
                     for x in sorted(kept, key=lambda x: (-x["grade"], x["passage_id"])))
    return GoldItem(qid=qid, qtype=record["qtype"], question=decision.get("question", record["question"]),
                    answer=decision.get("answer", record["answer"]), evidence=evidence, styles=tuple(record["styles"]),
                    original_question=record["original_question"], seed_docs=tuple(record["seed_docs"]))


def check_item(item: GoldItem, passage_text: dict[str, str]) -> None:
    """Инварианты набора: у негативного вопроса нет релевантных отрывков, у остальных — есть;
    у каждого размеченного отрывка есть дословная цитата из него."""
    has_relevant = bool(item.relevant_docs())
    if (item.qtype == "negative") == has_relevant:
        raise ValueError(f"{item.qid}: тип {item.qtype}, релевантных документов {len(item.relevant_docs())}")
    unquoted = [e.passage_id for e in item.evidence if not contains_quote(passage_text[e.passage_id], e.quote)]
    if unquoted:
        raise ValueError(f"{item.qid}: нет дословной цитаты у {unquoted}")


def check_review_digests(decisions: dict[str, dict], records: dict[str, dict]) -> None:
    """Решения ревью относятся к тому же содержимому кандидатов, которое просматривал ревьюер."""
    stale = [cid for cid, d in decisions.items() if d.get("digest") != review_digest(records[cid])]
    if stale:
        raise ValueError(f"содержимое кандидатов изменилось после ревью: {stale}")


def stage_finalize() -> None:
    docs, _ = load_corpus()
    passage_text = {p.passage_id: p.text for p in corpus_passages(docs.values())}
    records = {r["cid"]: r for r in read_stage("labeled")}
    decisions = json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
    check_review_digests(decisions, records)
    accepted = [cid for cid in sorted(decisions) if decisions[cid]["decision"] == "accept"]
    per_type = Counter(records[cid]["qtype"] for cid in accepted)
    if per_type != Counter(QUOTAS):
        raise ValueError(f"принятых по типам {dict(per_type)}, нужно {QUOTAS}")
    ordered = sorted(accepted, key=lambda cid: (list(QUOTAS).index(records[cid]["qtype"]), cid))
    items = [to_gold_item(records[cid], decisions[cid], f"q{i:02d}", passage_text)
             for i, cid in enumerate(ordered, start=1)]
    for item in items:
        check_item(item, passage_text)
    split = stratified_split(items, TEST_SHARE, SEED)
    items = [replace(i, split=split[i.qid]) for i in items]
    save_gold(items, GOLD_DIR / "gold.jsonl")
    print(Counter((i.qtype, i.split) for i in items))
    print(Counter(d["decision"] for d in decisions.values()))


# ---------- отчёт о стоимости ----------

def report_usage(stage: str, responses) -> None:
    fresh = [r for r in responses if not r.cached]
    tokens_in, tokens_out = sum(r.prompt_tokens for r in fresh), sum(r.completion_tokens for r in fresh)
    print(f"[{stage}] запросов {len(responses)}, новых {len(fresh)}, токенов вход/выход {tokens_in}/{tokens_out}")


STAGES = {"generate": stage_generate, "stylize": stage_stylize, "label": stage_label,
          "review": stage_review, "finalize": stage_finalize}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=STAGES)
    STAGES[ap.parse_args().stage]()
