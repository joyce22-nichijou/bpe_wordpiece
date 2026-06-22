"""
train_tokenizer.py
===================
Train BPE / WordPiece tokenizers (fast mode only) with on-disk caching.

CLI (non-interactive, all parameters passed as flags):
    python train_tokenizer.py --algo bpe --corpus wikitext --size large --vocab-size 10000
    python train_tokenizer.py --algo wp  --corpus gutenberg --size small --vocab-size 4000 --min-frequency 100

Flags:
    --algo          bpe / wp
    --corpus        wikitext / gutenberg (see load_data.CORPORA)
    --size          small / medium / large (see load_data.CORPUS_SIZES;
                    20% / 50% / 100% of the training corpus's sentences)
    --vocab-size    target vocabulary size
    --min-frequency required for wp; rejected for bpe
    --force-retrain ignore any cached vocab file and retrain

Vocab files are cached per algorithm (see vocab_path):
    vocab/bpe_vocab/bpe_vocab_{corpus}_{size}_v{vocab_size}.json
    vocab/wp_vocab/wp_vocab_{corpus}_{size}_v{vocab_size}_minfreq{N}.json

Import for reuse (e.g. from run_eval.py), which loads from cache instead
of retraining whenever a matching vocab file already exists:
    from train_tokenizer import get_or_train
    tok = get_or_train("bpe", "wikitext", "large", 10000)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from bpe import BPETokenizer
from wordpiece import WordPieceTokenizer
from load_data import CORPORA, CORPUS_SIZES, load_train_corpus

VOCAB_DIR = Path("vocab")
BPE_DIR   = VOCAB_DIR / "bpe_vocab"
WP_DIR    = VOCAB_DIR / "wp_vocab"

ALGO_CLASSES = {"bpe": BPETokenizer, "wp": WordPieceTokenizer}

SAMPLE_WORDS = ["running", "unhappiness", "international", "tokenization",
                "antidisestablishmentarianism"]


# ─────────────────────────────────────────
# Naming convention + JSON read/write
# ─────────────────────────────────────────

def vocab_path(algo: str, corpus_name: str, size: str, vocab_size: int,
               min_frequency: int | None = None) -> Path:
    if algo == "bpe":
        return BPE_DIR / f"bpe_vocab_{corpus_name}_{size}_v{vocab_size}.json"
    if algo == "wp":
        mf = min_frequency if min_frequency is not None else 1
        return WP_DIR / f"wp_vocab_{corpus_name}_{size}_v{vocab_size}_minfreq{mf}.json"
    raise ValueError(f"Unknown algo: {algo!r}, choose from {sorted(ALGO_CLASSES)}")


def save_tokenizer(tok, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"vocab": sorted(tok.vocab), "merges": [list(p) for p in tok.merges]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_tokenizer(algo: str, path: Path):
    tok = ALGO_CLASSES[algo]()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    tok.vocab = set(data["vocab"])
    tok.merges = [tuple(pair) for pair in data.get("merges", [])]
    tok.is_trained = True
    return tok


# ─────────────────────────────────────────
# get_or_train: the shared cache-aware entry point
# ─────────────────────────────────────────

def get_or_train(algo: str, corpus_name: str, size: str, vocab_size: int,
                  min_frequency: int = 1, force_retrain: bool = False):
    """Load the cached tokenizer for this config if it exists, else train and save it."""
    if algo not in ALGO_CLASSES:
        raise ValueError(f"Unknown algo: {algo!r}, choose from {sorted(ALGO_CLASSES)}")
    if corpus_name not in CORPORA:
        raise ValueError(f"Unknown corpus_name: {corpus_name!r}, choose from {sorted(CORPORA)}")
    if size not in CORPUS_SIZES:
        raise ValueError(f"Unknown size: {size!r}, choose from {sorted(CORPUS_SIZES)}")

    path = vocab_path(algo, corpus_name, size, vocab_size,
                       min_frequency if algo == "wp" else None)

    if path.exists() and not force_retrain:
        print(f"  [cache hit] {path} exists, loading (no retraining).")
        return load_tokenizer(algo, path)

    extra = f", min_frequency={min_frequency}" if algo == "wp" else ""
    print(f"  [cache miss] training a new {algo.upper()} tokenizer "
          f"(corpus={corpus_name}, size={size}, vocab_size={vocab_size}{extra}) ...")
    corpus = load_train_corpus(corpus_name, size)

    tok = ALGO_CLASSES[algo]()
    if algo == "bpe":
        tok.train(corpus, vocab_size, fast=True)
    else:
        tok.train(corpus, vocab_size, min_frequency=min_frequency, fast=True)

    save_tokenizer(tok, path)
    print(f"  trained and saved -> {path}")
    return tok


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a BPE or WordPiece tokenizer (fast mode), caching the result under vocab/.",
    )
    p.add_argument("--algo", required=True, choices=sorted(ALGO_CLASSES),
                   help="bpe or wp")
    p.add_argument("--corpus", required=True, choices=sorted(CORPORA),
                   help="training corpus source")
    p.add_argument("--size", required=True, choices=sorted(CORPUS_SIZES),
                   help="training corpus scale: small(20%%) / medium(50%%) / large(100%%)")
    p.add_argument("--vocab-size", required=True, type=int, dest="vocab_size",
                   help="target vocabulary size")
    p.add_argument("--min-frequency", type=int, default=None, dest="min_frequency",
                   help="required for wp; rejected for bpe")
    p.add_argument("--force-retrain", action="store_true",
                   help="ignore any cached vocab file and retrain")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.algo == "wp" and args.min_frequency is None:
        build_arg_parser().error("--algo wp requires --min-frequency")
    if args.algo == "bpe" and args.min_frequency is not None:
        build_arg_parser().error("--algo bpe does not accept --min-frequency")

    tok = get_or_train(
        args.algo, args.corpus, args.size, args.vocab_size,
        min_frequency=args.min_frequency if args.min_frequency is not None else 1,
        force_retrain=args.force_retrain,
    )

    print(f"\n  vocab_size={len(tok.vocab)}  merges={len(tok.merges)}")
    print("  sample tokenizations:")
    for w in SAMPLE_WORDS:
        print(f"    {w!r:>32} -> {tok.tokenize(w)}")


if __name__ == "__main__":
    import sys
    # Windows consoles often default to GBK/cp936, which can't print all
    # output here; force UTF-8.
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")
    main()
