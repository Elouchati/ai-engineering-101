# Ask the handbook — a RAG chatbot over your own documents

**Episode 6** of [AI Engineering 101](https://www.youtube.com/@techinonemin1).

A complete retrieval-augmented generation pipeline in three Python files:
chunking, embeddings, a vector store, retrieval, the prompt, and a Streamlit UI.

The point of it is the last part: **when the answer isn't in your documents, it
says so.** A chatbot that is right 90% of the time and confident 100% of the
time is worse than no chatbot, because nobody can tell those two states apart.

---

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # add your key
python ingest.py              # builds store.pkl from docs/
streamlit run app.py
```

The first run downloads the embedding model (~90 MB). After that, embeddings are
local, free and offline — only the final answer needs an API key.

Point it at your own material by dropping `.md` or `.txt` files into `docs/` and
re-running `python ingest.py`.

---

## What's in each file

| | |
|---|---|
| `rag.py` | The whole pipeline. No Streamlit import anywhere in it, so you can drive it from a plain REPL — which is where you'll actually debug it. |
| `ingest.py` | Reads `docs/`, chunks, embeds, writes `store.pkl`. Run once per change to your documents. |
| `app.py` | ~60 lines of Streamlit. |
| `eval.py` | **Episode 7** — twelve labelled questions and a scorer. Run `python eval.py` to get hit@k and top-1. |
| `docs/handbook.md` | The sample corpus both videos use. Keep it if you want to reproduce the numbers below. |

```python
from rag import Store, ask
store = Store.load("store.pkl")
print(ask(store, "is delivery free?"))
```

---

## The bug the episode is built around

The first version of `chunk_text()` split on blank lines and nothing else. That
let a single chunk start in the Delivery section and finish inside Returns — and
a chunk that spans two topics is about neither of them.

Asking *"is delivery free?"* then returned the **wrong** section first:

| | split on blank lines | split on headings |
|---|---|---|
| Returns | **0.462** ← top hit | 0.330 |
| Delivery | 0.455 | **0.472** ← top hit |
| gap | 0.007 | 0.142 |

Seven thousandths is a coin toss. The fix is three lines — treat a heading as a
hard boundary and never let a chunk cross it — and it widened the gap twenty
times over, with no prompt change, no bigger model and no re-ranker.

**Where you cut the document beats every other knob in the system.** Where a
document tells you its own structure — headings, sections, articles, clauses —
split there first, then count words.

You can reproduce both columns: the buggy chunker is just `re.split(r"\n\s*\n")`
with the `_HEADING` pass removed.

---

## Two other things worth knowing

**This is not keyword search.** Ask *"when does the shop shut?"* — neither
"shop" nor "shut" appears anywhere in `handbook.md`, so a keyword search returns
nothing. This returns the warehouse opening hours at 0.578.

**The vector store is 30 lines of numpy, on purpose.** Every vector is unit
length, so cosine similarity is a dot product and searching the whole store is
one matrix multiply. That is genuinely what a vector database does — it adds an
index so this stays fast at millions of rows instead of thousands. Under roughly
50k chunks you don't need one; over it, `search()` is the only function you
replace, and its signature is the same in Chroma, pgvector and Pinecone.

---

## Tuning

Two knobs matter far more than the rest:

- **`CHUNK_WORDS`** (default 120). Try 60 and 250 and watch the scores move.
  Too big buries the fact you needed; too small cuts a thought in half.
- **`k`** — how many passages go into the prompt. There's a slider for it in the
  UI, deliberately, because a constant buried in a file never gets touched.

One caveat: this reads text and markdown. A PDF with two columns or scanned
pages needs real extraction first — bad extraction is bad chunking, and bad
chunking is bad retrieval.

---

## Measuring it — `eval.py`

Episode 7 is built on this file. Twelve questions, each labelled with the
section that should answer it, worded the way a customer would type them rather
than the way the handbook is written:

```bash
python eval.py        # hit@4 100%, top-1 83%
python eval.py 1      # hit@1  83%
```

Two numbers come out. **hit@k** — was the right section anywhere in the k
passages retrieved? **top-1** — was it the very first one? hit@k is the one that
matters, since the model reads all k; top-1 tells you whether the ranking is
actually good.

That distinction is not academic. The blank-line chunking bug above scores
100% on hit@4 either way — it is only visible in top-1. If you evaluate by
eyeballing final answers, you will never find it.

Measured on this eval set:

| Broken how | hit@4 | top-1 |
|---|---|---|
| nothing — the working system | 100% | 83% |
| chunks split on blank lines | 100% | **58%** |
| k = 1 | **83%** | 83% |
| questions embedded by a different model | **92%** | **58%** |

Twelve questions is enough to start. Write yours before you tune anything, and
word them the way your users talk.

---

## Licence

MIT.
