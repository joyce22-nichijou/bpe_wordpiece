"""
wordpiece.py
============
Member B's implementation: Naive WordPiece + Fast WordPiece.
Both inherit from tokenizer_interface.BaseTokenizer.

Key differences from BPE (bpe.py):
  1. Preprocessing: non-initial characters get a "##" prefix instead of
     appending "</w>".  "low" -> "l ##o ##w"  (vs BPE's "l o w </w>")
  2. Scoring:  score(A, B) = freq(AB) / (freq(A) * freq(B))
     rather than BPE's raw freq(AB).
  3. Merge: new token = A + B[2:] when B starts with "##", else A + B.
     The "##" on B is absorbed; A's prefix (if any) is kept.
  4. Inference: greedy longest-match on self.vocab (standard WordPiece / BERT).
     Non-initial pieces carry "##"; unknown words become "[UNK]".
"""

from __future__ import annotations
import heapq
import re
from collections import defaultdict, Counter

from tokenizer_interface import (
    BaseTokenizer,
    Vocab,
    TokenList,
    Corpus,
    MergeRule,
)


# ─────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────

def wp_preprocess(corpus: Corpus, lowercase: bool = True) -> dict[str, int]:
    """
    WordPiece-style preprocessing (replaces BPE's preprocess()).

    Non-initial characters of each word are prefixed with "##" to mark
    their continuation role.  No end-of-word marker is added.

    "low"    -> "l ##o ##w"        (BPE would give "l o w </w>")
    "newest" -> "n ##e ##w ##e ##s ##t"

    Args:
        corpus: list of raw sentences
    Returns:
        word_freq: {"l ##o ##w": 3, "n ##e ##w ##e ##s ##t": 2, ...}
    """
    word_freq: dict[str, int] = defaultdict(int)
    for sentence in corpus:
        if lowercase:
            sentence = sentence.lower()
        sentence = re.sub(r"[^a-z\s]", " ", sentence)
        for word in sentence.split():
            chars = list(word)
            if not chars:
                continue
            wp_chars = [chars[0]] + ["##" + c for c in chars[1:]]
            word_freq[" ".join(wp_chars)] += 1
    return dict(word_freq)


def get_token_freq(word_freq: dict[str, int]) -> dict[str, int]:
    """
    Count total occurrences of every token, weighted by word frequency.

    Returns:
        {"l": 4, "##o": 4, "##w": 5, ...}
    """
    token_freq: dict[str, int] = defaultdict(int)
    for word, freq in word_freq.items():
        for tok in word.split():
            token_freq[tok] += freq
    return dict(token_freq)


def get_pair_freq(word_freq: dict[str, int]) -> dict[tuple[str, str], int]:
    """
    Count adjacent-pair frequencies, weighted by word frequency.
    Functionally identical to BPE's get_stats(); separated here so that
    get_wp_scores() can call it directly.
    """
    pair_freq: dict[tuple[str, str], int] = defaultdict(int)
    for word, freq in word_freq.items():
        tokens = word.split()
        for i in range(len(tokens) - 1):
            pair_freq[(tokens[i], tokens[i + 1])] += freq
    return dict(pair_freq)


def get_wp_scores(
    token_freq: dict[str, int],
    pair_freq: dict[tuple[str, str], int],
) -> dict[tuple[str, str], float]:
    """
    Compute the WordPiece score for every adjacent pair:

        score(A, B) = freq(AB) / (freq(A) * freq(B))

    BPE would simply return freq(AB).  The WP normalisation favours pairs
    that almost always co-occur rather than pairs that are merely frequent
    because their individual pieces appear everywhere.

    Returns:
        scores: {(A, B): float, ...}
    """
    scores: dict[tuple[str, str], float] = {}
    for (a, b), ab_freq in pair_freq.items():
        fa = token_freq.get(a, 0)
        fb = token_freq.get(b, 0)
        if fa > 0 and fb > 0:
            scores[(a, b)] = ab_freq / (fa * fb)
    return scores


def wp_merge_vocab(
    pair: tuple[str, str],
    word_freq: dict[str, int],
) -> dict[str, int]:
    """
    Merge every occurrence of adjacent pair (A, B) in word_freq.

    New token = A + B[2:]  when B starts with "##"
              = A + B      otherwise

    Examples
    --------
    ("l",   "##o")  ->  "lo"     A=initial, drops ## from B
    ("##e", "##s")  ->  "##es"   A=continuation, keeps ## from A, drops from B
    ("##s", "##t")  ->  "##st"

    Compare with BPE's merge_vocab where new token is always A + B.
    """
    a, b = pair
    new_tok = a + (b[2:] if b.startswith("##") else b)
    new_word_freq: dict[str, int] = {}
    for word, freq in word_freq.items():
        tokens = word.split()
        merged: list[str] = []
        i = 0
        n = len(tokens)
        while i < n:
            if i < n - 1 and tokens[i] == a and tokens[i + 1] == b:
                merged.append(new_tok)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        new_word_freq[" ".join(merged)] = freq
    return new_word_freq


# ─────────────────────────────────────────
# WordPiece Tokenizer
# ─────────────────────────────────────────

class WordPieceTokenizer(BaseTokenizer):
    """
    WordPiece tokenizer.
        self.vocab:  set[str]          vocabulary learned during training
        self.merges: list[(str,str)]   ordered merge rules

    The public interface is identical to BPETokenizer.
    """

    def __init__(self):
        super().__init__()
        self.merges: list[MergeRule] = []

    # ─────────────────────────────────────────
    # Training entry point
    # ─────────────────────────────────────────
    def train(
        self,
        corpus: Corpus,
        vocab_size: int,
        *,
        min_frequency: int = 1,
        fast: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Train WordPiece on corpus until vocabulary reaches vocab_size.

        min_frequency: a pair is only eligible for merging while its
            (weighted) occurrence count is >= min_frequency. Training stops
            early, before reaching vocab_size, once no eligible pair remains.

        fast=True  -> heapq + incremental updates  (~O(n log n))
        fast=False -> naive: recomputes all scores from scratch each round

        Both modes produce identical self.vocab and self.merges.
        """
        if fast:
            self._train_fast(corpus, vocab_size, min_frequency=min_frequency, verbose=verbose)
        else:
            self._train_naive(corpus, vocab_size, min_frequency=min_frequency, verbose=verbose)

    # ── Naive training ──────────────────────────────────────────────────
    def _train_naive(self, corpus: Corpus, vocab_size: int, min_frequency: int = 1, verbose: bool = False) -> None:
        word_freq = wp_preprocess(corpus)

        self.vocab = set()
        for word in word_freq:
            for tok in word.split():
                self.vocab.add(tok)
        self.merges = []

        if verbose:
            print(f"[WP naive init] vocab_size={len(self.vocab)}")

        while len(self.vocab) < vocab_size:
            token_freq = get_token_freq(word_freq)
            pair_freq  = get_pair_freq(word_freq)
            if not pair_freq:
                break
            scores = get_wp_scores(token_freq, pair_freq)
            # Pairs occurring fewer than min_frequency times are not eligible.
            scores = {p: s for p, s in scores.items() if pair_freq[p] >= min_frequency}
            if not scores:
                break

            # Highest score wins; ties broken by lexicographically smallest pair
            # (mirrors BPE's tie-breaking so heap and naive stay in sync).
            best_pair  = min(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            best_score = scores[best_pair]

            word_freq  = wp_merge_vocab(best_pair, word_freq)
            a, b       = best_pair
            new_token  = a + (b[2:] if b.startswith("##") else b)
            self.vocab.add(new_token)
            self.merges.append(best_pair)

            if verbose:
                print(f"[WP naive merge {len(self.merges):>3}] "
                      f"{best_pair} (score={best_score:.6f}) -> {new_token!r}")

        self.is_trained = True

    # ── Fast training ────────────────────────────────────────────────────
    def _train_fast(self, corpus: Corpus, vocab_size: int, min_frequency: int = 1, verbose: bool = False) -> None:
        """
        Same structural optimisation as BPE's _train_fast:
            words[i]       mutable token list for word i
            freqs[i]       word frequency
            pair_freq[p]   live pair count
            pair_where[p]  word indices that contain pair p
            heap           lazy min-heap of (-score, pair, pf_snap, tfa_snap, tfb_snap)

        WordPiece extension:
            token_freq[t]  live count of token t across all words
            token_pairs[t] set of live pairs whose score involves t;
                           used to find pairs that need re-scoring when
                           token_freq[t] changes (denominator side-effect)

        Heap validation at pop time: ALL THREE snapshots (pair_freq, freq_A,
        freq_B) must match current values.  A mismatch on any snapshot means
        the score is stale — skip and pop again.  This handles both the
        numerator (pair_freq) and denominator (token_freq) going stale.

        min_frequency: an otherwise-valid entry whose pf_snap < min_frequency
        is also discarded (not eligible for merging); training stops once the
        heap is exhausted without finding an eligible pair.
        """
        word_freq = wp_preprocess(corpus)

        words: list[list[str]] = []
        freqs: list[int]       = []
        self.vocab = set()
        for w, f in word_freq.items():
            toks = w.split()
            words.append(toks)
            freqs.append(f)
            self.vocab.update(toks)
        self.merges = []

        # ── Initialise live data structures ──
        pair_freq:    dict[tuple[str, str], int]            = defaultdict(int)
        pair_where:   dict[tuple[str, str], set[int]]       = defaultdict(set)
        token_freq:   dict[str, int]                        = defaultdict(int)
        token_pairs:  dict[str, set[tuple[str, str]]]       = defaultdict(set)

        for wi, toks in enumerate(words):
            f = freqs[wi]
            for tok in toks:
                token_freq[tok] += f
            for a, b in zip(toks, toks[1:]):
                p = (a, b)
                pair_freq[p]  += f
                pair_where[p].add(wi)

        for p in list(pair_freq):
            a, b = p
            token_pairs[a].add(p)
            token_pairs[b].add(p)

        # ── Heap helpers ──
        def push_score(p: tuple[str, str]) -> None:
            a, b  = p
            pf    = pair_freq.get(p, 0)
            tfa   = token_freq.get(a, 0)
            tfb   = token_freq.get(b, 0)
            if pf > 0 and tfa > 0 and tfb > 0:
                heapq.heappush(heap, (-pf / (tfa * tfb), p, pf, tfa, tfb))

        heap: list = []
        for p in pair_freq:
            push_score(p)
        heapq.heapify(heap)

        if verbose:
            print(f"[WP fast init] vocab_size={len(self.vocab)}, pairs={len(pair_freq)}")

        # ── Main loop ──
        while len(self.vocab) < vocab_size:

            # (1) Pop best pair; skip stale entries.
            best_pair = None
            while heap:
                neg_score, p, pf_snap, tfa_snap, tfb_snap = heapq.heappop(heap)
                a, b = p
                if (pair_freq.get(p, 0)    == pf_snap  and
                        token_freq.get(a, 0) == tfa_snap and
                        token_freq.get(b, 0) == tfb_snap and
                        pf_snap >= min_frequency):
                    best_pair = p
                    break
            if best_pair is None:
                break

            a, b      = best_pair
            best_score = pair_freq[best_pair] / (token_freq[a] * token_freq[b])
            new_tok    = a + (b[2:] if b.startswith("##") else b)
            self.vocab.add(new_tok)
            self.merges.append(best_pair)

            # (2) Remove merged pair from live structures.
            affected = pair_where.pop(best_pair, set())
            pair_freq.pop(best_pair, None)
            token_pairs[a].discard(best_pair)
            token_pairs[b].discard(best_pair)

            # (3) Update affected words; collect pairs whose score changed.
            score_refresh: set[tuple[str, str]] = set()

            for wi in affected:
                f        = freqs[wi]
                old_toks = words[wi]

                old_pair_counts = Counter(zip(old_toks, old_toks[1:]))
                old_tok_counts  = Counter(old_toks)

                new_toks = self._apply_one_merge(old_toks, best_pair)
                words[wi] = new_toks

                new_pair_counts = Counter(zip(new_toks, new_toks[1:]))
                new_tok_counts  = Counter(new_toks)

                # Diff-update pair_freq and pair_where.
                for p in set(old_pair_counts) | set(new_pair_counts):
                    pa, pb = p
                    delta_p = (new_pair_counts.get(p, 0) - old_pair_counts.get(p, 0)) * f
                    if delta_p != 0:
                        pair_freq[p] = pair_freq.get(p, 0) + delta_p
                        score_refresh.add(p)
                    if new_pair_counts.get(p, 0) > 0:
                        pair_where[p].add(wi)
                        token_pairs[pa].add(p)
                        token_pairs[pb].add(p)
                    else:
                        pair_where[p].discard(wi)

                # Diff-update token_freq.
                for t in set(old_tok_counts) | set(new_tok_counts):
                    delta_t = (new_tok_counts.get(t, 0) - old_tok_counts.get(t, 0)) * f
                    if delta_t != 0:
                        token_freq[t] = token_freq.get(t, 0) + delta_t
                        if token_freq.get(t, 0) <= 0:
                            token_freq.pop(t, None)
                        # token_freq[t] changed -> all pairs involving t need
                        # their score refreshed (denominator side effect).
                        score_refresh |= set(token_pairs.get(t, set()))

            # (4) Push fresh scores for all changed pairs.
            for p in score_refresh:
                if pair_freq.get(p, 0) > 0:
                    push_score(p)

            if verbose:
                print(f"[WP fast merge {len(self.merges):>3}] "
                      f"{best_pair} (score={best_score:.6f}) -> {new_tok!r}")

        self.is_trained = True

    # ─────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────
    def tokenize(self, text: str) -> TokenList:
        """
        Tokenize text using greedy longest-match (standard WordPiece inference).

        For each word, scan left-to-right and greedily pick the longest prefix
        present in self.vocab.  Non-initial segments are looked up with the
        "##" prefix.  If any position has no matching prefix (even a single
        character), the entire word is replaced by "[UNK]".

        This fixes two issues with the previous BPE-style approach:
          - merge-order inference could miss valid vocab entries
          - unknown characters produced partial [UNK]s inside a word;
            now the whole word becomes [UNK] (standard BERT behaviour)
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer is not trained yet; call train() first.")

        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        all_tokens: TokenList = []
        for word in text.split():
            all_tokens.extend(self._tokenize_word(word))
        return all_tokens

    def _tokenize_word(self, word: str) -> TokenList:
        """Greedy longest-match for a single word; returns ['[UNK]'] on failure."""
        tokens: TokenList = []
        start = 0
        n = len(word)
        while start < n:
            end = n
            cur_substr = None
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                return ["[UNK]"]
            tokens.append(cur_substr)
            start = end
        return tokens

    @staticmethod
    def _apply_one_merge(tokens: TokenList, pair: tuple[str, str]) -> TokenList:
        """
        Apply one merge rule left-to-right.

        new token = A + B[2:]  if B starts with "##"
                  = A + B      otherwise
        (Same control flow as BPE; only the token-formation differs.)
        """
        a, b     = pair
        new_tok  = a + (b[2:] if b.startswith("##") else b)
        i        = 0
        merged: TokenList = []
        n = len(tokens)
        while i < n:
            if i < n - 1 and tokens[i] == a and tokens[i + 1] == b:
                merged.append(new_tok)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged



# ─────────────────────────────────────────
# Self-tests
# ─────────────────────────────────────────

if __name__ == "__main__":

    import json
    import sys
    import time
    from datetime import datetime
    from pathlib import Path

    # Project layout: this script lives in src/, data/vocab/results are siblings.
    PROJECT_ROOT      = Path(__file__).resolve().parent.parent
    DATA_DIR          = PROJECT_ROOT / "data"
    WP_VOCAB_DIR      = PROJECT_ROOT / "vocab" / "wp_vocab"
    TRAIN_RESULTS_DIR = PROJECT_ROOT / "results" / "train_results"

    # ── Hyperparameters ──────────────────────────────────────────────────────
    # corpus_name: "gutenberg" | "wikitext103"
    # train_mode:  "fast" | "naive" | "both"
    corpus_name   = "wikitext103"
    train_mode    = "fast"
    target_vocab  = 10000
    min_frequency = 500
    n_books       = 600          # only used when corpus_name == "gutenberg"

    # ── Corpus loaders (cache to project directory on first run) ─────────────

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

    # ── Load corpus ──────────────────────────────────────────────────────────
    if corpus_name == "gutenberg":
        real_corpus = load_corpus_gutenberg(n_books)
        corpus_desc = f"Project Gutenberg (sedthh/gutenberg_english, {n_books} books)"
        corpus_tag  = f"gutenberg{n_books}books"
    elif corpus_name == "wikitext103":
        real_corpus = load_corpus_wikitext103()
        corpus_desc = "WikiText-103 (wikitext-103-raw-v1, train split)"
        corpus_tag  = "wikitext103"
    else:
        raise ValueError(f"Unknown corpus_name: {corpus_name!r}. Choose 'gutenberg' or 'wikitext103'.")

    print("=" * 60)
    print(f"WordPiece training  |  corpus={corpus_name}  |  "
          f"vocab_size={target_vocab}  |  min_frequency={min_frequency}")
    print("=" * 60)

    print(f"Corpus size in memory: {sys.getsizeof(real_corpus) / 1024 / 1024:.1f} MB  |  "
          f"estimated text size: {sum(len(l) for l in real_corpus) / 1024 / 1024:.1f} MB")

    word_freq_g = wp_preprocess(real_corpus)
    print(f"Lines: {len(real_corpus)}  |  "
          f"unique words after wp_preprocess: {len(word_freq_g)}  |  "
          f"total word occurrences: {sum(word_freq_g.values())}")

    # ── Train ────────────────────────────────────────────────────────────────
    timing: dict[str, float] = {}

    def _run(fast: bool) -> WordPieceTokenizer:
        label = "fast" if fast else "naive"
        print(f"[{label}] training started...")
        t0 = time.perf_counter()
        wp = WordPieceTokenizer()
        wp.train(real_corpus, vocab_size=target_vocab, min_frequency=min_frequency,
                 fast=fast, verbose=False)
        elapsed = time.perf_counter() - t0
        timing[label] = elapsed
        print(f"[{label}] vocab={len(wp.vocab)}  merges={len(wp.merges)}  time={elapsed:.2f}s")
        return wp

    if train_mode == "fast":
        wp_real = _run(fast=True)
    elif train_mode == "naive":
        wp_real = _run(fast=False)
    elif train_mode == "both":
        wp_naive = _run(fast=False)
        wp_real  = _run(fast=True)
        print(f"speedup (naive/fast): {timing['naive'] / timing['fast']:.1f}x")
    else:
        raise ValueError(f"Unknown train_mode: {train_mode!r}. Choose 'fast', 'naive', or 'both'.")

    long_toks = sorted(wp_real.vocab, key=len, reverse=True)[:15]
    print(f"Longest tokens (top 15): {long_toks}")

    test_words = ["running", "landlord", "sleeping", "whale", "cannibal", "unknown",
                  "playing", "national", "university", "international", "revolutionary",
                  "extraordinary", "unbelievable", "preprocessing", "tokenization",
                  "anabaptist", "counterrevolutionary", "antidisestablishmentarianism"]

    print("\nSample tokenizations:")
    for w in test_words:
        print(f"  {w!r:>30} -> {wp_real.tokenize(w)}")

    # The saved vocab corresponds to wp_real: the fast run for "fast"/"both",
    # the naive run for "naive". The fast/naive tag is appended after minfreq.
    vocab_mode = "naive" if train_mode == "naive" else "fast"

    # ── 保存词表文件（供 wordpiece_test.py 加载）────────────────────────────
    WP_VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    vocab_path = WP_VOCAB_DIR / f"wp_vocab_{corpus_name}_v{target_vocab}_minfreq{min_frequency}_{vocab_mode}.json"
    vocab_data = {
        "vocab":  sorted(wp_real.vocab),
        "merges": [list(pair) for pair in wp_real.merges],
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    print(f"\nVocab saved to: {vocab_path}")

    # ── 导出训练结果（带时间戳，记录本次运行统计）───────────────────────────
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "corpus":         corpus_desc,
        "corpus_lines":   len(real_corpus),
        "unique_words":   len(word_freq_g),
        "total_tokens":   sum(word_freq_g.values()),
        "train_mode":     train_mode,
        "training_time_seconds": {k: round(v, 3) for k, v in timing.items()},
        "vocab_size":     len(wp_real.vocab),
        "merges_learned": len(wp_real.merges),
        "min_frequency":  min_frequency,
        "longest_tokens": long_toks,
        "sample_tokenizations": {
            w: wp_real.tokenize(w) for w in test_words
        },
    }

    TRAIN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAIN_RESULTS_DIR / f"wp_results_{corpus_tag}_vocab{target_vocab}_minfreq{min_frequency}_{vocab_mode}_{timestamp_str}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {out_path}")