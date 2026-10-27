# gigaschool-rag

Сквозной RAG-проект курса «LLM-инженер» (GigaSchool): один «грязный» корпус и серия
экспериментов над ним — от сбора золотого набора до генерации и оценки ответов.

| Шаг | Отчёт | Ноутбуки |
|---|---|---|
| КТ 1 — корпус и золотой набор | [`reports/kt1.md`](reports/kt1.md) | `kt1_eda`, `kt1_gold` |
| КТ 2 — предобработка, чанкинг, метаданные | [`reports/kt2.md`](reports/kt2.md) | `kt2_chunking` |
| КТ 3 — векторная БД, эмбеддинги, sparse/dense/hybrid | [`reports/kt3.md`](reports/kt3.md) | `kt3_search` |
| КТ 4 — трансформация запросов, реранкинг | — | — |
| КТ 5 — генерация, оценка, продвинутый RAG | — | — |
| Финальный отчёт | — | — |

План и чек-листы — [PLAN.md](PLAN.md).

## Код

```
src/gigarag/     общая библиотека: регулярки (patterns.py), текст, шум, дубликаты, чанкинг, метрики, кэш, источники
scripts/         тонкие CLI всех КТ, settings.py — общие пути и параметры
notebooks/       ноутбуки КТ (исходник — .py в формате jupytext, .ipynb — исполненный)
reports/         отчёты КТ: решения, таблицы, выводы
gold/            золотой набор
data/            сырьё, корпус, кэши (вне git, воспроизводятся скриптами)
tests/           pytest на функции библиотеки
```

Правила кода и воспроизводимости — [CLAUDE.md](CLAUDE.md).

```bash
uv sync                                   # окружение (Linux — PyTorch с CUDA 12.8, иначе PyPI)
uv run ruff check src tests scripts notebooks
uv run pytest
```
