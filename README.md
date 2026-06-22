# BPE & WordPiece from scratch

BPE and WordPiece subword tokenizers implemented from scratch in pure Python
(stdlib only — no HuggingFace `tokenizers`), plus an evaluation pipeline that
trains both on real corpora and scores them against gold-standard morphology
data (MorphoLex, Morpho Challenge GoldStd).


## Project structure

```
tokenizer_interface.py   shared types + BaseTokenizer interface (used by both tokenizers)
bpe.py                   BPE tokenizer (Naive + Fast)
wordpiece.py             WordPiece tokenizer (Naive + Fast)
train_tokenizer.py       CLI: train or load a cached tokenizer
load_data.py             loads corpora + gold-standard morphology data
metrics.py               metric computations (fertility, UNK rate, boundary P/R/F1, ...)
run_eval.py              interactive menu — main entry point for results

data/        corpora + gold-standard files (auto-downloaded/cached)
vocab/       trained tokenizers, cached as JSON (bpe_vocab/, wp_vocab/)
results/     metrics_summary.xlsx (the durable results table) + tokenization caches
```

Do not `git add -A`/`git add .` — `data/` and `vocab/` contain large files that
should be staged by name only.

## How to run

**Train a tokenizer:**
```bash
python train_tokenizer.py --algo bpe --corpus wikitext --size large --vocab-size 10000
python train_tokenizer.py --algo wp  --corpus wikitext --size large --vocab-size 10000 --min-frequency 100
```
`--algo` is `bpe`/`wp`, `--corpus` is `wikitext`/`gutenberg`, `--size` is
`small`(20%)/`medium`(50%)/`large`(100%) of that corpus. `--min-frequency` is
required for `wp`, rejected for `bpe`. Run `python train_tokenizer.py --help`
for the full option list. Results are cached under `vocab/`, so re-running the
same config loads instead of retraining.

**Produce evaluation results** (the main program — no flags, fully interactive):
```bash
python run_eval.py
```
It prints a menu with 4 choices, prompting for whatever parameters each one
needs (corpus / size / vocab_size / min_frequency):
1. **Evaluate one tokenizer** — basic stats + morphological metrics for one
   (algo, corpus, size, vocab_size) config.
2. **Compare BPE vs WordPiece vs gold standard** — same metrics, side by side,
   plus sample-word segmentations.
3. **Case study** — type any word/sentence to see how BPE / WordPiece / the gold
   standards segment it (exploration only, nothing is saved).
4. Exit.

Every metric from options 1–2 is written to `results/metrics_summary.xlsx`
(re-running a config updates its row rather than duplicating it).

**Other:**
```bash
python load_data.py [path/to/goldstd/file]   # parser self-check, not a results program
```
`python bpe.py` / `python wordpiece.py` run a full real-corpus training in their
`__main__` block (minutes, not a quick test) — for a fast sanity check, import
the class instead: `BPETokenizer().train(small_corpus, vocab_size=...)`.
