"""Сборка сырого корпуса КТ 1 из скачанных источников в data/corpus.jsonl.

Ничего не чистим: HTML-разметка статей и комментариев, эмодзи и ссылки
в постах Telegram остаются как есть — чистка и её эффект измеряются в КТ 2.
Решения по отбору (окно патчей, офтоп Telegram) описаны в README КТ 1.
"""

from collections import Counter

from gigarag.corpus import Document, ensure_unique_ids, write_jsonl
from gigarag.sources import DocumentSource
from gigarag.sources.dota2ru import Dota2ruSource
from gigarag.sources.patchnotes import PatchnotesSource, entity_vocabulary, patch_dates
from gigarag.sources.telegram import TelegramSource
from gigarag.sources.topic_lexicon import build_topic_matcher
from settings import CORPUS, OFFTOPIC_SHARE, RAW_DOTA2RU, RAW_PATCHES, RAW_TELEGRAM, SEED, SINCE_PATCH


def corpus_sources() -> list[DocumentSource]:
    """Источники в порядке записи в корпус."""
    dates = patch_dates(RAW_PATCHES)
    return [
        PatchnotesSource(RAW_PATCHES, SINCE_PATCH),
        Dota2ruSource(RAW_DOTA2RU, SINCE_PATCH, dates),
        TelegramSource(RAW_TELEGRAM, since_date=dates[SINCE_PATCH],
                       is_topical=build_topic_matcher(entity_vocabulary(RAW_PATCHES)),
                       offtopic_share=OFFTOPIC_SHARE, seed=SEED),
    ]


def collect(sources: list[DocumentSource]) -> list[Document]:
    return ensure_unique_ids([doc for source in sources for doc in source.documents()])


def report(docs: list[Document], digest: str) -> None:
    for source, n in Counter(d.source for d in docs).items():
        print(f"{source:18} {n}")
    print("всего:", len(docs), "→", CORPUS)
    print("sha256:", digest)


def main() -> None:
    docs = collect(corpus_sources())
    report(docs, write_jsonl(docs, CORPUS))


if __name__ == "__main__":
    main()
