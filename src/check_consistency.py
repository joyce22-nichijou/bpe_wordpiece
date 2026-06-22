"""
Check whether the naive and fast WordPiece implementations produce
identical vocabularies and merges on the same training run.

Usage:
    python check_consistency.py
"""

import json
from pathlib import Path

# This script lives in src/; vocab files live in ../vocab/wp_vocab/.
WP_VOCAB_DIR = Path(__file__).resolve().parent.parent / "vocab" / "wp_vocab"
NAIVE_PATH = WP_VOCAB_DIR / "wp_vocab_wikitext103_v10000_minfreq500_naive.json"
FAST_PATH  = WP_VOCAB_DIR / "wp_vocab_wikitext103_v10000_minfreq500_fast.json"


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    for p in (NAIVE_PATH, FAST_PATH):
        if not Path(p).exists():
            print(f"File not found: {p}")
            return

    naive = load(NAIVE_PATH)
    fast  = load(FAST_PATH)

    # Whole-file equality
    identical = (naive == fast)
    print(f"Full JSON identical: {identical}")

    if identical:
        return

    # If not identical, break it down so you can see where they diverge
    print("\nField-by-field comparison:")
    keys = sorted(set(naive.keys()) | set(fast.keys()))
    for k in keys:
        if k not in naive:
            print(f"  [{k}] missing in naive")
            continue
        if k not in fast:
            print(f"  [{k}] missing in fast")
            continue
        same = (naive[k] == fast[k])
        print(f"  [{k}] equal: {same}")

        if not same and k in ("vocab", "merges"):
            n_set = set(map(tuple, naive[k])) if isinstance(naive[k], list) else set(naive[k])
            f_set = set(map(tuple, fast[k]))  if isinstance(fast[k],  list) else set(fast[k])
            only_naive = n_set - f_set
            only_fast  = f_set - n_set
            print(f"      only in naive ({len(only_naive)}): "
                  f"{list(only_naive)[:5]}{'...' if len(only_naive) > 5 else ''}")
            print(f"      only in fast  ({len(only_fast)}): "
                  f"{list(only_fast)[:5]}{'...' if len(only_fast) > 5 else ''}")


if __name__ == "__main__":
    main()
