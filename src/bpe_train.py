"""
bpe_train.py
============
Train a BPE tokenizer on a real corpus and save both:
  1. bpe_vocab_{corpus_name}_v{vocab_size}.json
     -> the model: {"vocab": [...], "merges": [[a,b], ...]}
  2. bpe_results_{corpus_tag}_vocab{vocab_size}_{ts}.json
     -> the training statistics

Note: BPE deliberately has no min_frequency parameter (cf. bpe.py docstring).
B's wordpiece.py keeps min_frequency in its own CLI because WordPiece's
score-ratio mechanism is unsafe without an absolute-frequency floor.

Usage
-----
  python bpe_train.py wikitext103 10000
  python bpe_train.py wikitext103 20000
  python bpe_train.py gutenberg   10000 --n-books 600
  python bpe_train.py gutenberg   10000 --n-books 1200
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from bpe import BPETokenizer
from tokenizer_interface import preprocess


# ─────────────────────────────────────────
# Project layout: this script lives in src/, data/vocab/results are siblings.
# ─────────────────────────────────────────
PROJECT_ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR          = PROJECT_ROOT / "data"
BPE_VOCAB_DIR     = PROJECT_ROOT / "vocab" / "bpe_vocab"
TRAIN_RESULTS_DIR = PROJECT_ROOT / "results" / "train_results"


# ─────────────────────────────────────────
# Corpus loaders
# ─────────────────────────────────────────

def load_corpus_gutenberg(n_books: int) -> list[str]:
    cache_path = DATA_DIR / f"corpus_gutenberg_{n_books}books.txt"
    if cache_path.exists():
        print(f"Loading Gutenberg corpus from local cache: {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f if line.strip()]
    print(f"Downloading {n_books} books from HuggingFace (sedthh/gutenberg_english, streaming)...")
    from datasets import load_dataset
    ds = load_dataset("sedthh/gutenberg_english", split="train", streaming=True)
    lines: list[str] = []
    for i, book in enumerate(ds):
        if i >= n_books:
            break
        for line in book["TEXT"].split("\n"):
            line = line.strip()
            if line:
                lines.append(line)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved to: {cache_path}  ({len(lines)} lines, "
          f"{cache_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return lines


def load_corpus_wikitext103() -> list[str]:
    cache_path = DATA_DIR / "corpus_wikitext103.txt"

    def _keep(s: str) -> bool:
        s = s.strip()
        return bool(s) and not s.startswith("=")

    if cache_path.exists():
        print(f"Loading WikiText-103 corpus from local cache: {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f if _keep(line)]
    print("Downloading WikiText-103 train split from HuggingFace (wikitext-103-raw-v1)...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    lines = [row["text"].strip() for row in ds if _keep(row["text"])]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved to: {cache_path}  ({len(lines)} lines, "
          f"{cache_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return lines


SAMPLE_WORDS = [
    "running", "landlord", "sleeping", "whale", "cannibal", "unknown",
    "playing", "national", "university", "international", "revolutionary",
    "extraordinary", "unbelievable", "preprocessing", "tokenization",
    "anabaptist", "counterrevolutionary", "antidisestablishmentarianism",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train BPE on a real corpus and save vocab + results JSON."
    )
    parser.add_argument("corpus", choices=["wikitext103", "gutenberg"])
    parser.add_argument("vocab_size", type=int,
                        help="Target vocabulary size (e.g. 10000, 20000).")
    parser.add_argument("--n-books", type=int, default=None,
                        help="For gutenberg: number of books (e.g. 600, 1200).")
    parser.add_argument("--naive", action="store_true",
                        help="Use the Naive O(n^2) trainer instead of Fast.")
    parser.add_argument("--no-save-vocab", action="store_true",
                        help="Skip writing the vocab JSON (useful for quick tests).")
    parser.add_argument("--results-dir", default=str(TRAIN_RESULTS_DIR),
                        help="Directory to write bpe_results_*.json into.")
    parser.add_argument("--progress-every", type=int, default=100,
                        help="Print a progress update every N merges (0 = silent).")
    args = parser.parse_args()

    # ── Resolve corpus identifiers ────────────────────────────────────
    if args.corpus == "wikitext103":
        corpus_name = "wikitext103"
        corpus_tag  = "wikitext103"
        corpus_desc = "WikiText-103 (wikitext-103-raw-v1, train split)"
        real_corpus = load_corpus_wikitext103()
    elif args.corpus == "gutenberg":
        if args.n_books is None:
            print("Error: --n-books is required when corpus=gutenberg", file=sys.stderr)
            sys.exit(2)
        n = args.n_books
        corpus_name = f"gutenberg{n}"
        corpus_tag  = f"gutenberg{n}books"
        corpus_desc = f"Project Gutenberg (sedthh/gutenberg_english, {n} books)"
        real_corpus = load_corpus_gutenberg(n)
    else:
        raise ValueError(args.corpus)

    target_vocab = args.vocab_size
    train_mode   = "naive" if args.naive else "fast"

    print("=" * 60)
    print(f"BPE training  |  corpus={corpus_name}  |  "
          f"vocab_size={target_vocab}  |  mode={train_mode}")
    print("=" * 60)
    print(f"Corpus size in memory: {sys.getsizeof(real_corpus) / 1024 / 1024:.1f} MB  |  "
          f"estimated text size: {sum(len(l) for l in real_corpus) / 1024 / 1024:.1f} MB")

    # ── Preprocess (for reporting stats) ──────────────────────────────
    t_pre = time.perf_counter()
    word_freq = preprocess(real_corpus)
    t_pre = time.perf_counter() - t_pre
    unique_words = len(word_freq)
    total_tokens = sum(word_freq.values())
    print(f"Lines: {len(real_corpus)}  |  unique words: {unique_words}  |  "
          f"total occurrences: {total_tokens}  |  preprocess: {t_pre:.1f}s")

    # ── Train ────────────────────────────────────────────────────────
    print(f"[{train_mode}] training started...")
    t0 = time.perf_counter()
    bpe = BPETokenizer()
    bpe.train(real_corpus, vocab_size=target_vocab,
              fast=not args.naive, verbose=False, progress_every=args.progress_every)
    elapsed = time.perf_counter() - t0
    print(f"[{train_mode}] vocab={len(bpe.vocab)}  merges={len(bpe.merges)}  time={elapsed:.2f}s")

    long_toks = sorted(bpe.vocab, key=len, reverse=True)[:15]
    print(f"Longest tokens (top 15): {long_toks}")

    print("\nSample tokenizations:")
    for w in SAMPLE_WORDS:
        print(f"  {w!r:>30} -> {bpe.tokenize(w)}")

    # ── Save vocab JSON ───────────────────────────────────────────────
    if not args.no_save_vocab:
        BPE_VOCAB_DIR.mkdir(parents=True, exist_ok=True)
        vocab_path = BPE_VOCAB_DIR / f"bpe_vocab_{corpus_name}_v{target_vocab}.json"
        vocab_data = {
            "algorithm": "BPE",
            "vocab":     sorted(bpe.vocab),
            "merges":    [list(pair) for pair in bpe.merges],
        }
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"\nVocab saved to: {vocab_path}")

    # ── Save results JSON ─────────────────────────────────────────────
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm":      "BPE",
        "corpus":         corpus_desc,
        "corpus_lines":   len(real_corpus),
        "unique_words":   unique_words,
        "total_tokens":   total_tokens,
        "train_mode":     train_mode,
        "training_time_seconds": {train_mode: round(elapsed, 3)},
        "vocab_size":     len(bpe.vocab),
        "merges_learned": len(bpe.merges),
        "longest_tokens": long_toks,
        "sample_tokenizations": {w: bpe.tokenize(w) for w in SAMPLE_WORDS},
    }
    out_path = results_dir / (
        f"bpe_results_{corpus_tag}_vocab{target_vocab}_{timestamp_str}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
