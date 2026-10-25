# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # КТ 1 — EDA сырого корпуса
#
# Корпус собран скриптом `scripts/build_corpus.py` (описание источников и решений — в `README.md`).
# Здесь он анализируется **до какой-либо чистки**: HTML-разметка, эмодзи, ссылки и дубликаты на месте.
#
# Вопросы, на которые отвечает EDA:
# 1. Какой типичный размер документа и насколько он варьируется?
# 2. Что составляет основной шум в данных?
# 3. Какие тематические кластеры есть в корпусе?
# 4. Какие проблемы это создаст для чанкинга и ретрива?

# %%
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import emoji
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

pd.set_option("display.max_colwidth", 120)
plt.rcParams["figure.dpi"] = 110

ROOT = Path.cwd()
DATA = ROOT / "data"
CACHE = DATA / "cache"
CACHE.mkdir(exist_ok=True)

df = pd.DataFrame([json.loads(l) for l in open(DATA / "corpus.jsonl", encoding="utf-8")])
df["date"] = pd.to_datetime(df["date"])
TAG_RE = re.compile(r"<[^>]+>")
df["visible"] = df["text"].map(lambda t: re.sub(r"\s+", " ", TAG_RE.sub(" ", t)).strip())
print(len(df), "документов")

# %% [markdown]
# ## 1. Состав корпуса

# %%
by_src = df.groupby("source").agg(docs=("doc_id", "size"), first=("date", "min"), last=("date", "max"))
by_src["share_%"] = (by_src["docs"] / len(df) * 100).round(1)
by_src

# %%
monthly = df.groupby([df["date"].dt.to_period("M"), "source"]).size().unstack(fill_value=0)
ax = monthly.plot(kind="bar", stacked=True, figsize=(13, 4), width=0.9)
ax.set_xticks(range(0, len(monthly), 3))
ax.set_xticklabels([str(p) for p in monthly.index[::3]], rotation=45)
ax.set_title("Документы по месяцам и источникам")
ax.set_ylabel("документов")
plt.tight_layout()

# %% [markdown]
# ## 2. Длины документов
#
# Токены считаются токенизатором **BGE-M3** (XLM-RoBERTa, SentencePiece) — это одна из моделей,
# которыми корпус будет индексироваться дальше, так что длины близки к реальным.
# Считаем два варианта: сырой текст (как его увидел бы наивный пайплайн, с HTML) и видимый текст без тегов.

# %%
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
len_cache = CACHE / "token_lens.parquet"
if len_cache.exists():
    lens = pd.read_parquet(len_cache)
else:
    def count(texts):
        return [len(x) for x in tok(list(texts), add_special_tokens=False)["input_ids"]]
    lens = pd.DataFrame({"doc_id": df["doc_id"], "tok_raw": count(df["text"]), "tok_visible": count(df["visible"])})
    lens.to_parquet(len_cache)
df = df.merge(lens, on="doc_id")
print("токенов всего: сырых", df["tok_raw"].sum(), "| видимых", df["tok_visible"].sum())

# %%
def length_stats(col):
    g = df.groupby("source")[col]
    s = g.describe(percentiles=[0.1, 0.5, 0.9])[["count", "mean", "std", "min", "10%", "50%", "90%", "max"]]
    s["<50_%"] = g.apply(lambda x: (x < 50).mean() * 100)
    s[">2000_%"] = g.apply(lambda x: (x > 2000).mean() * 100)
    s.loc["ВСЕГО"] = [*df[col].describe(percentiles=[0.1, 0.5, 0.9])[["count", "mean", "std", "min", "10%", "50%", "90%", "max"]],
                     (df[col] < 50).mean() * 100, (df[col] > 2000).mean() * 100]
    return s.round(1)

print("Видимый текст:")
display(length_stats("tok_visible"))
print("Сырой текст с разметкой:")
display(length_stats("tok_raw"))

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
bins = np.logspace(0, np.log10(df["tok_raw"].max() + 1), 60)
for ax, col, title in zip(axes, ["tok_visible", "tok_raw"], ["видимый текст", "сырой текст с HTML"]):
    for src, g in df.groupby("source"):
        ax.hist(g[col].clip(lower=1), bins=bins, alpha=0.55, label=src)
    ax.set_xscale("log")
    ax.axvline(50, color="k", ls="--", lw=1)
    ax.axvline(2000, color="k", ls="--", lw=1)
    ax.set_title(f"Распределение длин, {title}")
    ax.set_xlabel("токенов BGE-M3 (лог. шкала)")
axes[0].legend()
plt.tight_layout()

# %% [markdown]
# Примеры выбросов: самые короткие и самые длинные документы.

# %%
cols = ["source", "tok_visible", "visible"]
display(df[df["tok_visible"] < 5].sample(8, random_state=1)[cols])
display(df.nlargest(5, "tok_visible")[["source", "tok_visible", "tok_raw", "title"]])

# %% [markdown]
# ## 3. Шум: доля «мусорных» токенов
#
# Сырой текст разбивается регулярным выражением на единицы и каждая относится к классу:
# HTML-тег, URL, эмодзи, число, спецсимвол или слово. Доля классов показывает,
# насколько агрессивной должна быть предобработка в КТ 2.
#
# URL вынимаются первым проходом — в том числе из атрибутов `href`/`src`, иначе
# они растворились бы внутри HTML-тегов; эмодзи — вторым, через библиотеку `emoji`.

# %%
URL_RE = re.compile(r"https?://[^\s\"'<>]+|www\.[^\s\"'<>]+|t\.me/[^\s\"'<>]+")
UNIT_RE = re.compile(r"(?P<html><[^>]+>)|(?P<num>\d+(?:[.,]\d+)*%?)|(?P<word>[^\W\d_]+)|(?P<special>[^\w\s])")

def noise_profile(text):
    c = Counter()
    text, c["url"] = URL_RE.subn(" ", text)
    text = emoji.replace_emoji(text, replace=" \x00 ")
    c["emoji"] = text.count("\x00")
    for m in UNIT_RE.finditer(text.replace("\x00", " ")):
        c[m.lastgroup] += 1
    return c

noise = pd.DataFrame([noise_profile(t) for t in df["text"]]).fillna(0)
noise["source"] = df["source"].values
cls = ["word", "html", "url", "emoji", "num", "special"]
share = noise.groupby("source")[cls].sum()
share.loc["ВСЕГО"] = share.sum()
share = (share.div(share.sum(axis=1), axis=0) * 100).round(1)
share["мусор_%"] = (100 - share["word"]).round(1)
share

# %%
share.drop(columns="мусор_%").plot(kind="barh", stacked=True, figsize=(11, 4), colormap="tab20")
plt.title("Состав сырого текста по классам единиц, %")
plt.xlabel("%")
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()

# %% [markdown]
# Дополнительные признаки «грязи» по источникам: доля документов с эмодзи, ссылками, цитатами,
# пересылками, медиа без текста, а также рекламные маркеры в Telegram.

# %%
# «бонус» сюда не входит: в патчноутах это игровой термин («бонус к урону»)
ADS_RE = re.compile(r"промокод|ставк|казино|букмекер|фрибет|реклама|erid|бездеп", re.I)
flags = pd.DataFrame({
    "source": df["source"],
    "с эмодзи": noise["emoji"] > 0,
    "со ссылкой": noise["url"] > 0,
    "с цитатой": df["meta"].map(lambda m: bool(m.get("has_quote"))),
    "пересылка": df["meta"].map(lambda m: bool(m.get("forwarded_from"))),
    "медиа": df["meta"].map(lambda m: bool(m.get("has_photo") or m.get("has_video"))),
    "реклама/ставки": df["visible"].str.contains(ADS_RE),
    "латиница>50%": df["visible"].map(lambda t: sum(ch.isascii() and ch.isalpha() for ch in t) > 0.5 * max(1, sum(ch.isalpha() for ch in t))),
})
(flags.groupby("source").mean() * 100).round(1)

# %% [markdown]
# ## 4. Дубликаты и near-duplicates
#
# Только фиксируем, ничего не удаляем (удаление — шаг предобработки КТ 2).
# * **Точные** — совпадение видимого текста после приведения к нижнему регистру и схлопывания пробелов.
# * **Почти-дубликаты** — MinHash (128 перестановок) по словесным 3-граммам, LSH с порогом Jaccard ≥ 0.8;
#   документы короче 5 слов не участвуют — у них шинглов почти нет.

# %%
norm = df["visible"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
h = norm.map(lambda t: hashlib.md5(t.encode()).hexdigest())
dup_groups = h[h.duplicated(keep=False)].groupby(h).size()
exact = pd.DataFrame({"source": df["source"], "exact_dup": h.duplicated(keep="first")})
print(f"Точные дубликаты: {exact['exact_dup'].sum()} лишних копий ({exact['exact_dup'].mean()*100:.1f}% корпуса), групп {len(dup_groups)}")
display(exact.groupby("source")["exact_dup"].agg(["sum", "mean"]).assign(mean=lambda x: (x["mean"] * 100).round(1)).rename(columns={"sum": "копий", "mean": "%"}))
top = df.loc[h.isin(dup_groups.nlargest(6).index)].groupby(h)["visible"].agg(["size", "first"]).sort_values("size", ascending=False)
top

# %%
mh_cache = CACHE / "minhash_pairs.json"
words = norm.str.findall(r"\w+")
if mh_cache.exists():
    pairs = json.loads(mh_cache.read_text())
else:
    lsh = MinHashLSH(threshold=0.8, num_perm=128)
    mhs = {}
    for i, w in enumerate(words):
        if len(w) < 5:
            continue
        m = MinHash(num_perm=128)
        m.update_batch([" ".join(w[j:j + 3]).encode() for j in range(len(w) - 2)])
        mhs[i] = m
        lsh.insert(str(i), m)
    pairs = sorted({tuple(sorted((i, int(j)))) for i, m in mhs.items() for j in lsh.query(m) if int(j) != i})
    mh_cache.write_text(json.dumps(pairs))

pairs = [p for p in pairs if h.iloc[p[0]] != h.iloc[p[1]]]  # точные уже посчитаны выше
near_ids = {i for p in pairs for i in p}
print(f"Пар почти-дубликатов (не точных): {len(pairs)}, затронуто документов: {len(near_ids)} ({len(near_ids)/len(df)*100:.1f}%)")
src = df["source"].values
pair_src = Counter(" ↔ ".join(sorted({src[a], src[b]})) if src[a] != src[b] else src[a] for a, b in pairs)
pd.Series(pair_src, name="пар").sort_values(ascending=False).to_frame()

# %%
rng = np.random.default_rng(0)
for a, b in [pairs[i] for i in rng.choice(len(pairs), size=min(4, len(pairs)), replace=False)]:
    print(f"[{src[a]}] {df['visible'].iloc[a][:200]}\n[{src[b]}] {df['visible'].iloc[b][:200]}\n" + "-" * 80)

# %% [markdown]
# ## 5. Тематические кластеры
#
# Видимый текст (первые 512 токенов) кодируется моделью `intfloat/multilingual-e5-small`
# на GPU, затем KMeans. Ключевые слова кластера — TF-IDF по «склеенному» тексту кластера,
# где слова, встречающиеся в большинстве кластеров, отсекаются (`max_df`).
# Это разведочный анализ: цель — увидеть, из каких тем состоит корпус, а не выбрать модель.

# %%
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

emb_cache = CACHE / "e5small_visible.npy"
if emb_cache.exists():
    emb = np.load(emb_cache)
else:
    model = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda")
    model.max_seq_length = 512
    emb = model.encode(["passage: " + t for t in df["visible"]], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    np.save(emb_cache, emb)

K = 14
df["cluster"] = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(emb)
STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня
еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя
ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтобы без будто чего раз тоже себе под будет ж тогда кто
этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при
наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою
этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между это the of to and in is for on""".split())
docs_by_cl = df.groupby("cluster")["visible"].apply(lambda s: " ".join(s.str.lower()))
vec = TfidfVectorizer(max_df=0.5, min_df=2, token_pattern=r"(?u)\b[^\W\d_]{3,}\b", stop_words=list(STOP), max_features=50000)
X = vec.fit_transform(docs_by_cl)
terms = np.array(vec.get_feature_names_out())
clusters = pd.DataFrame({
    "docs": df.groupby("cluster").size(),
    "top_terms": [", ".join(terms[np.argsort(-X[i].toarray()[0])[:10]]) for i in range(K)],
})
clusters = clusters.join((pd.crosstab(df["cluster"], df["source"], normalize="index") * 100).round(0))
clusters.sort_values("docs", ascending=False)

# %%
from sklearn.decomposition import PCA

xy = PCA(n_components=2, random_state=0).fit_transform(emb)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for src_name, g in df.groupby("source"):
    axes[0].scatter(xy[g.index, 0], xy[g.index, 1], s=2, alpha=0.4, label=src_name)
axes[0].legend(markerscale=6)
axes[0].set_title("PCA эмбеддингов: источники")
axes[1].scatter(xy[:, 0], xy[:, 1], c=df["cluster"], s=2, alpha=0.4, cmap="tab20")
for c in range(K):
    cx, cy = xy[df["cluster"] == c].mean(axis=0)
    axes[1].text(cx, cy, str(c), fontsize=11, weight="bold")
axes[1].set_title("PCA эмбеддингов: кластеры KMeans")
plt.tight_layout()

# %% [markdown]
# Примеры документов из каждого кластера:

# %%
for c in clusters.sort_values("docs", ascending=False).index:
    s = df[df["cluster"] == c].sample(2, random_state=0)
    print(f"--- кластер {c}: {clusters.loc[c, 'top_terms'][:80]}")
    for _, r in s.iterrows():
        print(f"  [{r['source']}] {r['visible'][:160]}")

# %% [markdown]
# ## 6. Сводка цифр для выводов

# %%
summary = {
    "документов": len(df),
    "токенов видимого текста (BGE-M3)": int(df["tok_visible"].sum()),
    "токенов сырого текста": int(df["tok_raw"].sum()),
    "медиана длины, видимый": int(df["tok_visible"].median()),
    "доля < 50 токенов, %": round((df["tok_visible"] < 50).mean() * 100, 1),
    "доля > 2000 токенов, %": round((df["tok_visible"] > 2000).mean() * 100, 1),
    "мусорных единиц в сыром тексте, %": share.loc["ВСЕГО", "мусор_%"],
    "точных дубликатов, %": round(exact["exact_dup"].mean() * 100, 1),
    "документов в near-dup парах, %": round(len(near_ids) / len(df) * 100, 1),
}
pd.Series(summary).to_frame("значение")

# %% [markdown]
# ## 7. Выводы
#
# **1. Типичный размер документа и разброс.** Распределение резко неоднородное, с тяжёлым хвостом:
# медиана — 58 токенов видимого текста, но стандартное отклонение ~720.
# * Комментарии dota2.ru — медиана 21 токен, 84% короче 50 токенов (часто это реплика из пары слов: «+1», «Почему?»).
# * Посты Telegram — медиана 57–79 токенов, длинных почти нет (максимум < 1000).
# * Блоки патчноутов — медиана 54, но 46% короче 50 (одна правка одного предмета) и редкие крупные блоки
#   (масштабные патчи 7.36/7.38/7.41 — до 2,2 тыс. токенов на героя).
# * Статьи dota2.ru — всего 36 документов (0,2% корпуса), но 16% всех токенов: медиана 4,1 тыс.,
#   максимум 55 тыс. токенов; 97% длиннее 2000. Это полные пересказы патчноутов.
#
# В итоге 40% документов короче 50 токенов, а 0,2% (все — статьи) длиннее 2000.
#
# **2. Основной шум.** В сыром виде только 41% единиц текста — слова.
# * HTML — главный шум: 38% единиц, в Telegram до 47% (`<i class="emoji">`, `<b>`, `<a href>`, `<br>`).
#   Из-за разметки сырой корпус в BGE-M3-токенах в 5 раз больше видимого (10,6 млн против 2,1 млн).
# * Спецсимволы — 13%, эмодзи — 3% единиц, но эмодзи есть в 99,5% постов Telegram (заголовки-рубрики).
# * Ссылки — в 92–94% постов Telegram (подпись канала, промо-ссылки), сами по себе ~1% единиц.
# * Числа — 16% единиц в патчноутах, но это не шум, а суть («урон увеличен с 75 до 80»):
#   при чистке их удалять нельзя.
# * Шаблонный текст: подписи каналов («Gabe News | Dota 2», «@Dota2»), рубрики-шаблоны
#   («Еженедельный мета-отчёт»), удалённые и пустые комментарии (только картинка или цитата).
# * Дубликаты: 2,2% лишних точных копий (больше всего — шаблонные посты @Dota2 и «Данный комментарий был удалён»),
#   ещё 1,7% документов — почти-дубликаты, в основном отредактированные и перепощенные посты одного канала.
# * Реклама ставок оказалась редкой (~1% постов) — каналы отфильтрованы по теме.
#
# **3. Тематические кластеры.** Эмбеддинги разделяют корпус прежде всего по **стилю источника**,
# а уже внутри — по теме:
# * изменения способностей и героев (кластеры 1, 5 — патчноуты);
# * пересказы патчей в Telegram (13) — те же факты, но другим языком;
# * турниры и результаты матчей (7); составы команд, трансферы, интервью (3);
# * мета и статистика: винрейты, пикрейты, «сильнейшие герои» (2, 12);
# * советы, билды, механики (6); косметика, сокровищницы, мастерская (0); мемы, хайлайты, косплей (4);
# * обновления в процессе: staging-сервер, утечки, Deadlock (8);
# * реакции форума — жалобы на баланс, короткие эмоции, сленг (9, 10, 11).
#
# **4. Проблемы для чанкинга и ретрива.**
# * **Один размер чанка не подходит никому:** статьи надо резать, а 40% документов и так короче 50 токенов —
#   фиксированный чанк в 512 токенов для них бессмыслен, а короткие комментарии без контекста статьи
#   («что с ним») почти не несут информации — понадобятся метаданные (к какому патчу/статье относится).
# * **Без удаления HTML** embedding-окно наполовину заполнено разметкой, а BM25 индексирует атрибуты тегов.
# * **SentenceChunker** будет ошибаться на постах Telegram: предложения там разделены эмодзи, а не точками.
# * **Шаблонные посты и дубликаты** будут занимать несколько мест в top-k одинаковым текстом.
# * **Один факт — несколько источников:** правка есть в патчноуте, в статье, в пересказе Telegram и
#   в обсуждении. В gold-наборе нужно указывать все релевантные документы, иначе Recall занизится.
# * **Разные формы имён:** в патчноутах (русская локализация Valve) имена героев и предметов английские
#   (Pudge, Black King Bar), в постах и комментариях — русские и сленговые (Пудж, бкб).
#   BM25 на этом будет проигрывать, dense-модели должны быть мультиязычными.
# * **Временная неоднозначность:** одна и та же способность меняется в нескольких патчах 7.34–7.41,
#   поэтому вопросы без указания патча неоднозначны, а фильтрация по метаданным (патч, дата) должна помогать.
