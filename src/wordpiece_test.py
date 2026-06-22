"""
wordpiece_test.py
=================
Load a saved WordPiece vocab file and tokenize a test set.

Modes
-----
1. Cross-domain mode – run all 4 vocab × test-corpus combinations and export JSONs:
     python wordpiece_test.py cross <wiki_corpus> <wiki_vocab_size> <wiki_minfreq>
                                    <guten_corpus> <guten_vocab_size> <guten_minfreq>
                                    [output_dir]
   Example:
     python wordpiece_test.py cross wikitext103 10000 500 gutenberg1200 10000 500
     python wordpiece_test.py cross wikitext103 10000 500 gutenberg1200 10000 500 results/

   4 output files produced (one per combination):
     [same  domain]  wiki vocab  × wiki test
     [cross domain]  wiki vocab  × guten test
     [cross domain]  guten vocab × wiki test
     [same  domain]  guten vocab × guten test

2. Single mode (original behaviour):
     python wordpiece_test.py <corpus> <vocab_size> <min_frequency> [txt_file]
   Example:
     python wordpiece_test.py wikitext103 10000 500
     python wordpiece_test.py gutenberg1200 10000 500 my_words.txt

Vocab file naming convention (fast/naive tag appended after minfreq):
     wp_vocab_{corpus}_v{vocab_size}_minfreq{min_frequency}_{mode}.json

Output JSON naming convention:
     wp_cross_{vocab_corpus}_v{vocab_size}_minfreq{minfreq}_{mode}__{test_label}_{timestamp}.json
"""

from __future__ import annotations
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from wordpiece import WordPieceTokenizer


# ─────────────────────────────────────────
# Project layout: this script lives in src/, data/vocab/results are siblings.
# ─────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR         = PROJECT_ROOT / "data"
WP_VOCAB_DIR     = PROJECT_ROOT / "vocab" / "wp_vocab"
TEST_RESULTS_DIR = PROJECT_ROOT / "results" / "test_results"

# fast and naive training produce identical vocabularies, so tests load the
# "fast" vocab by default; change this to "naive" to load the naive ones.
VOCAB_MODE = "fast"


# ── Default sample words ──────────────────────────────────────────────────────

SAMPLE_WORDS = [
    "running", "landlord", "sleeping", "whale", "cannibal", "unknown",
    "playing", "national", "university", "international", "revolutionary",
    "extraordinary", "unbelievable", "preprocessing", "tokenization",
    "anabaptist", "counterrevolutionary", "antidisestablishmentarianism",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_vocab_file(corpus: str, vocab_size: int, min_frequency: int,
                    mode: str = VOCAB_MODE) -> dict:
    path = WP_VOCAB_DIR / f"wp_vocab_{corpus}_v{vocab_size}_minfreq{min_frequency}_{mode}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Vocab file not found: {path.resolve()}\n"
            f"Run wordpiece.py with corpus={corpus!r}, vocab_size={vocab_size}, "
            f"min_frequency={min_frequency} (mode={mode!r}) to generate it."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_tokenizer(vocab_data: dict) -> WordPieceTokenizer:
    wp = WordPieceTokenizer()
    wp.vocab = set(vocab_data["vocab"])
    wp.merges = [tuple(pair) for pair in vocab_data.get("merges", [])]
    wp.is_trained = True
    return wp


def words_from_txt(txt_path: str) -> list[str]:
    """Extract unique words (insertion order preserved) using the same preprocessing as wp_preprocess."""
    seen: dict[str, None] = {}
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.lower()
            line = re.sub(r"[^a-z\s]", " ", line)
            for word in line.split():
                if word and word not in seen:
                    seen[word] = None
    return list(seen)


def _fmt_mb(path: Path) -> str:
    return f"{path.stat().st_size / 1024 / 1024:.2f} MB"


# ── Test corpus loaders ───────────────────────────────────────────────────────

def get_test_corpus_wikitext103() -> tuple[list[str], str]:
    """Return (words, source_label) from the WikiText-103 test split."""
    local_path = DATA_DIR / "test_corpus_wikitext103.txt"

    if local_path.exists():
        print(f"  Loading WikiText-103 test corpus from local file: {local_path}  ({_fmt_mb(local_path)})")
        words = words_from_txt(str(local_path))
        print(f"    -> {len(words)} unique words")
        return words, "wikitext103_test"

    print("  Downloading WikiText-103 test split from HuggingFace …")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not found. Install it with: pip install datasets")
        sys.exit(1)

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n".join(row for row in ds["text"])
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved to: {local_path}  ({_fmt_mb(local_path)})")
    words = words_from_txt(str(local_path))
    print(f"    -> {len(words)} unique words")
    return words, "wikitext103_test"


def get_test_corpus_gutenberg() -> tuple[list[str], str]:
    """Return (words, source_label) from Project Gutenberg books (IDs 1201-1210)."""
    local_path = DATA_DIR / "test_corpus_gutenberg.txt"

    if local_path.exists():
        print(f"  Loading Gutenberg test corpus from local file: {local_path}  ({_fmt_mb(local_path)})")
        words = words_from_txt(str(local_path))
        print(f"    -> {len(words)} unique words")
        return words, "gutenberg_test"

    try:
        import requests
    except ImportError:
        print("Error: 'requests' package not found. Install it with: pip install requests")
        sys.exit(1)

    print("  Downloading Project Gutenberg books (IDs 1201–1210) …")
    parts: list[str] = []
    for book_id in range(1201, 1211):
        urls = [
            f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
            f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
            f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
        ]
        downloaded = False
        for url in urls:
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200:
                    print(f"    Book {book_id}: {len(r.content) / 1024:.1f} KB")
                    parts.append(r.text)
                    downloaded = True
                    break
            except requests.RequestException:
                continue
        if not downloaded:
            print(f"    Book {book_id}: download failed (skipped)")

    combined = "\n".join(parts)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"  Saved to: {local_path}  ({_fmt_mb(local_path)})")
    words = words_from_txt(str(local_path))
    print(f"    -> {len(words)} unique words")
    return words, "gutenberg_test"


# ── Core test runner ──────────────────────────────────────────────────────────

def _domain_type(vocab_corpus: str, test_label: str) -> str:
    """Determine whether this is a same-domain or cross-domain experiment."""
    wiki_vocab  = "wikitext" in vocab_corpus.lower()
    wiki_test   = "wikitext" in test_label.lower()
    guten_vocab = "gutenberg" in vocab_corpus.lower()
    guten_test  = "gutenberg" in test_label.lower()
    if (wiki_vocab and wiki_test) or (guten_vocab and guten_test):
        return "same"
    return "cross"


def run_tokenization_test(
    vocab_corpus: str,
    vocab_size: int,
    min_frequency: int,
    test_words: list[str],
    test_label: str,
    output_dir: Path,
    mode: str = VOCAB_MODE,
) -> Path:
    """Load vocab, tokenize test_words, write JSON to output_dir, return the output path."""
    vocab_filename = f"wp_vocab_{vocab_corpus}_v{vocab_size}_minfreq{min_frequency}_{mode}.json"
    try:
        vocab_data = load_vocab_file(vocab_corpus, vocab_size, min_frequency, mode)
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        sys.exit(1)

    wp = build_tokenizer(vocab_data)
    print(f"  Vocab loaded: {len(wp.vocab)} tokens, {len(wp.merges)} merges  ({vocab_filename})")
    print(f"  Tokenizing {len(test_words)} words …", end=" ", flush=True)

    tokenizations: dict[str, list[str]] = {word: wp.tokenize(word) for word in test_words}
    unk_count = sum(1 for toks in tokenizations.values() if toks == ["[UNK]"])
    print(f"done  (UNK rate: {unk_count}/{len(test_words)} = {unk_count/len(test_words)*100:.2f}%)")

    domain = _domain_type(vocab_corpus, test_label)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = (
        f"wp_cross_{vocab_corpus}_v{vocab_size}_minfreq{min_frequency}_{mode}"
        f"__{test_label}_{timestamp}.json"
    )
    out_path = output_dir / out_name

    results = {
        "metadata": {
            "vocab_corpus":    vocab_corpus,
            "vocab_file":      vocab_filename,
            "vocab_size":      vocab_size,
            "min_frequency":   min_frequency,
            "train_mode":      mode,
            "test_corpus":     test_label,
            "domain_type":     domain,
            "num_words_tested": len(test_words),
            "unk_count":       unk_count,
            "unk_rate":        round(unk_count / len(test_words), 6) if test_words else 0.0,
            "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "tokenizations": tokenizations,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"  Saved -> {out_path}")
    return out_path


# ── Cross-domain mode ─────────────────────────────────────────────────────────

def run_cross_domain(
    wiki_corpus: str,  wiki_vocab_size: int, wiki_minfreq: int,
    guten_corpus: str, guten_vocab_size: int, guten_minfreq: int,
    output_dir: Path,
    mode: str = VOCAB_MODE,
) -> None:
    """Run all 4 (vocab corpus × test corpus) combinations and export one JSON each."""
    print("=" * 68)
    print("Cross-domain WordPiece tokenization test")
    print(f"  Wiki  vocab : {wiki_corpus}  (size={wiki_vocab_size}, minfreq={wiki_minfreq})")
    print(f"  Guten vocab : {guten_corpus}  (size={guten_vocab_size}, minfreq={guten_minfreq})")
    print(f"  Output dir  : {output_dir.resolve()}")
    print("=" * 68)

    # Load both test corpora upfront (may be downloaded if missing)
    print("\n[1/2] Loading WikiText-103 test corpus …")
    wiki_words, wiki_label = get_test_corpus_wikitext103()

    print("\n[2/2] Loading Gutenberg test corpus …")
    guten_words, guten_label = get_test_corpus_gutenberg()

    combos = [
        # (vocab_corpus, vocab_size, minfreq, test_words, test_label)
        (wiki_corpus,  wiki_vocab_size,  wiki_minfreq,  wiki_words,  wiki_label),
        (wiki_corpus,  wiki_vocab_size,  wiki_minfreq,  guten_words, guten_label),
        (guten_corpus, guten_vocab_size, guten_minfreq, wiki_words,  wiki_label),
        (guten_corpus, guten_vocab_size, guten_minfreq, guten_words, guten_label),
    ]

    output_paths: list[Path] = []
    for idx, (vc, vs, vf, tw, tl) in enumerate(combos, 1):
        domain = _domain_type(vc, tl)
        print(f"\n[Experiment {idx}/4]  [{domain} domain]  vocab={vc}  test={tl}")
        out = run_tokenization_test(vc, vs, vf, tw, tl, output_dir, mode)
        output_paths.append(out)

    print("\n" + "=" * 68)
    print(f"All {len(output_paths)} result files saved:")
    for p in output_paths:
        print(f"  {p}")
    print("=" * 68)


# ── Single mode (original) ────────────────────────────────────────────────────

def run_single(args: list[str]) -> None:
    if len(args) < 3:
        print("Usage: python wordpiece_test.py <corpus> <vocab_size> <min_frequency> [txt_file]")
        sys.exit(1)

    corpus        = args[0]
    vocab_size    = int(args[1])
    min_frequency = int(args[2])
    txt_file      = args[3] if len(args) >= 4 else None
    mode          = VOCAB_MODE

    vocab_filename = f"wp_vocab_{corpus}_v{vocab_size}_minfreq{min_frequency}_{mode}.json"
    try:
        vocab_data = load_vocab_file(corpus, vocab_size, min_frequency, mode)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    wp = build_tokenizer(vocab_data)
    print(f"Loaded vocab: {len(wp.vocab)} tokens, {len(wp.merges)} merges  ({vocab_filename})")

    # Determine input words
    if txt_file is not None:
        txt_path = Path(txt_file)
        if not txt_path.exists():
            print(f"Error: txt file not found: {txt_path.resolve()}")
            sys.exit(1)
        words = words_from_txt(txt_file)
        source_label = txt_path.stem
        print(f"Input: {len(words)} unique words from {txt_file!r}")
    elif "wikitext" in corpus.lower():
        words, source_label = get_test_corpus_wikitext103()
        print(f"Input: {len(words)} unique words from WikiText-103 test set")
    elif "gutenberg" in corpus.lower():
        words, source_label = get_test_corpus_gutenberg()
        print(f"Input: {len(words)} unique words from Gutenberg books")
    else:
        words = SAMPLE_WORDS
        source_label = "sample"
        print(f"Input: {len(words)} built-in sample words")

    # Tokenize
    tokenizations: dict[str, list[str]] = {word: wp.tokenize(word) for word in words}
    unk_count = sum(1 for toks in tokenizations.values() if toks == ["[UNK]"])

    print("\nTokenizations:")
    for word, toks in tokenizations.items():
        print(f"  {word!r:>30} -> {toks}")

    print(f"\nUNK rate: {unk_count}/{len(words)} = {unk_count/len(words)*100:.2f}%" if words else "")

    # Save results
    TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TEST_RESULTS_DIR / f"wp_test_{corpus}_v{vocab_size}_minfreq{min_frequency}_{mode}_{source_label}.json"
    results = {
        "metadata": {
            "vocab_corpus":     corpus,
            "vocab_file":       vocab_filename,
            "vocab_size":       vocab_size,
            "min_frequency":    min_frequency,
            "train_mode":       mode,
            "input_source":     txt_file if txt_file is not None else source_label,
            "num_words_tested": len(words),
            "unk_count":        unk_count,
            "unk_rate":         round(unk_count / len(words), 6) if words else 0.0,
            "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "tokenizations": tokenizations,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "cross":
        args = sys.argv[2:]
        if len(args) < 6:
            print(
                "Usage: python wordpiece_test.py cross\n"
                "         <wiki_corpus>  <wiki_vocab_size>  <wiki_minfreq>\n"
                "         <guten_corpus> <guten_vocab_size> <guten_minfreq>\n"
                "         [output_dir]\n\n"
                "Example: python wordpiece_test.py cross "
                "wikitext103 10000 500 gutenberg1200 10000 500\n"
                "Example: python wordpiece_test.py cross "
                "wikitext103 10000 500 gutenberg1200 10000 500 results/"
            )
            sys.exit(1)

        wiki_corpus      = args[0]
        wiki_vocab_size  = int(args[1])
        wiki_minfreq     = int(args[2])
        guten_corpus     = args[3]
        guten_vocab_size = int(args[4])
        guten_minfreq    = int(args[5])
        output_dir       = Path(args[6]) if len(args) >= 7 else TEST_RESULTS_DIR

        run_cross_domain(
            wiki_corpus,  wiki_vocab_size,  wiki_minfreq,
            guten_corpus, guten_vocab_size, guten_minfreq,
            output_dir,
        )
    else:
        run_single(sys.argv[1:])


if __name__ == "__main__":
    main()
