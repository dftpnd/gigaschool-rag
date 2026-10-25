# gigaschool-rag

Сквозной RAG-проект курса «LLM-инженер» (GigaSchool): один «грязный» корпус и серия
экспериментов над ним — от сбора золотого набора до генерации и оценки ответов.

| Шаг | Папка |
|---|---|
| КТ 1 — корпус и золотой набор | [`kt1-corpus-gold/`](kt1-corpus-gold/) |
| КТ 2 — предобработка, чанкинг, метаданные | [`kt2-preprocessing-chunking/`](kt2-preprocessing-chunking/) |
| КТ 3 — векторная БД, эмбеддинги, sparse/dense/hybrid | [`kt3-vectordb-search/`](kt3-vectordb-search/) |
| КТ 4 — трансформация запросов, реранкинг | [`kt4-query-rerank/`](kt4-query-rerank/) |
| КТ 5 — генерация, оценка, продвинутый RAG | [`kt5-generation-eval/`](kt5-generation-eval/) |
| Финальный отчёт | [`final-report/`](final-report/) |

План и чек-листы — [PLAN.md](PLAN.md).

## Код

```
src/gigarag/     общая библиотека: регулярки (patterns.py), текст, шум, дубликаты, темы, кэш, источники корпуса
kt<N>-<тема>/    одна контрольная точка: README, ноутбуки, scripts/ (тонкие CLI), data/ (вне git)
tests/           pytest на функции библиотеки
```

Правила кода и воспроизводимости — [CLAUDE.md](CLAUDE.md).

```bash
uv sync                                   # окружение (Linux — PyTorch с CUDA 12.8, иначе PyPI)
uv run ruff check src tests kt*/scripts kt*/*.py
uv run pytest
```
