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
  4. Inference: same merge-order approach as BPE (not greedy longest-match).
     Output tokens carry "##" for non-initial pieces.
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
        fast: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Train WordPiece on corpus until vocabulary reaches vocab_size.

        fast=True  -> heapq + incremental updates  (~O(n log n))
        fast=False -> naive: recomputes all scores from scratch each round

        Both modes produce identical self.vocab and self.merges.
        """
        if fast:
            self._train_fast(corpus, vocab_size, verbose=verbose)
        else:
            self._train_naive(corpus, vocab_size, verbose=verbose)

    # ── Naive training ──────────────────────────────────────────────────
    def _train_naive(self, corpus: Corpus, vocab_size: int, verbose: bool = False) -> None:
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
    def _train_fast(self, corpus: Corpus, vocab_size: int, verbose: bool = False) -> None:
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
                        pf_snap > 0):
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
        text = re.sub(r"[^a-z\s]", "", text)
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

    def _apply_merges(self, tokens: TokenList) -> TokenList:
        """Apply every rule in self.merges to tokens, in order."""
        for pair in self.merges:
            tokens = self._apply_one_merge(tokens, pair)
        return tokens


# ─────────────────────────────────────────
# Self-tests
# ─────────────────────────────────────────

if __name__ == "__main__":

    # ── Block 1: wp_preprocess and scoring ──────────────────────────────
    print("=" * 60)
    print("Block 1: wp_preprocess + WP scoring")
    print("=" * 60)

    corpus = ["low low low lowest newest"]
    wf = wp_preprocess(corpus)
    print("wp_preprocess output:")
    for k, v in sorted(wf.items()):
        print(f"  {k!r}: {v}")

    tf = get_token_freq(wf)
    pf = get_pair_freq(wf)
    sc = get_wp_scores(tf, pf)

    print("\nToken frequencies:")
    for tok, cnt in sorted(tf.items()):
        print(f"  {tok!r}: {cnt}")

    print("\nPair scores (sorted descending):")
    for pair, score in sorted(sc.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {pair}: {score:.6f}  (pair_freq={pf[pair]}, "
              f"freq_A={tf[pair[0]]}, freq_B={tf[pair[1]]})")

    # Verify the score formula for one pair
    a, b = "##s", "##t"
    expected_score = pf[(a, b)] / (tf[a] * tf[b])
    assert abs(sc[(a, b)] - expected_score) < 1e-12, "score formula mismatch"
    print(f"\n[OK] score({(a,b)}) = {pf[(a,b)]}/({tf[a]}*{tf[b]}) = {expected_score:.6f}")

    # ── Block 2: Naive WP training ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Block 2: Naive WordPiece training")
    print("=" * 60)

    wp = WordPieceTokenizer()
    wp.train(corpus, vocab_size=15, fast=False, verbose=True)
    print(f"\nFinal vocab ({len(wp.vocab)}): {sorted(wp.vocab)}")
    print(f"merges ({len(wp.merges)}): {wp.merges}")

    # ── Block 3: tokenize ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Block 3: tokenize")
    print("=" * 60)
    for w in ["lowest", "newest", "low", "newer", "unknown"]:
        print(f"  tokenize({w!r:>10}) -> {wp.tokenize(w)}")

    # ── Block 4: Naive == Fast ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Block 4: Fast WordPiece — equivalence + benchmark")
    print("=" * 60)

    naive = WordPieceTokenizer(); naive.train(corpus, vocab_size=15, fast=False)
    fast  = WordPieceTokenizer(); fast.train(corpus,  vocab_size=15, fast=True)
    assert naive.vocab  == fast.vocab,  \
        f"vocab mismatch:\n  naive={sorted(naive.vocab)}\n  fast ={sorted(fast.vocab)}"
    assert naive.merges == fast.merges, \
        f"merges mismatch:\n  naive={naive.merges}\n  fast ={fast.merges}"
    print("[OK] small corpus (vocab_size=15): naive.vocab == fast.vocab "
          "and naive.merges == fast.merges")

    # ── Block 5: Speed benchmark ─────────────────────────────────────────
    import random, time
    random.seed(42)
    alphabet   = "abcdefghijklmnop"
    base_words = ["".join(random.choice(alphabet)
                          for _ in range(random.randint(3, 9)))
                  for _ in range(80)]
    sentences  = [" ".join(random.choice(base_words) for _ in range(20))
                  for _ in range(100)]
    medium_corpus = sentences
    target_vocab  = 300

    t0     = time.perf_counter()
    n2     = WordPieceTokenizer(); n2.train(medium_corpus, vocab_size=target_vocab, fast=False)
    t_naive = time.perf_counter() - t0

    t0    = time.perf_counter()
    f2    = WordPieceTokenizer(); f2.train(medium_corpus, vocab_size=target_vocab, fast=True)
    t_fast = time.perf_counter() - t0

    assert n2.vocab  == f2.vocab,  "medium-corpus vocab mismatch"
    assert n2.merges == f2.merges, "medium-corpus merges mismatch"
    speedup = t_naive / t_fast if t_fast > 0 else float("inf")
    print(f"\n  Medium corpus: {len(medium_corpus)} sentences x 20 words, "
          f"target_vocab={target_vocab}")
    print(f"  Naive: {t_naive*1000:7.1f} ms")
    print(f"  Fast : {t_fast*1000:7.1f} ms")
    print(f"  [OK] vocab/merges identical; Fast speedup ~ {speedup:.1f}x")

    sample = "abcfgh"
    print(f"\n  fast.tokenize({sample!r}) -> {f2.tokenize(sample)}")

    # ── Block 6: BPE vs WP on same corpus ────────────────────────────────
    print("\n" + "=" * 60)
    print("Block 6: BPE vs WordPiece comparison on same corpus")
    print("=" * 60)
    try:
        from bpe import BPETokenizer
        bpe_tok = BPETokenizer()
        bpe_tok.train(corpus, vocab_size=15, fast=False)
        wp_tok  = WordPieceTokenizer()
        wp_tok.train(corpus,  vocab_size=15, fast=False)
        print(f"{'word':>12}  {'BPE':30}  {'WordPiece':30}")
        print("-" * 78)
        for w in ["lowest", "newest", "low", "newer", "unknown"]:
            b = str(bpe_tok.tokenize(w))
            p = str(wp_tok.tokenize(w))
            print(f"  {w!r:>10}  {b:30}  {p:30}")
        print("\nBPE merges:       ", bpe_tok.merges)
        print("WordPiece merges: ", wp_tok.merges)
    except ImportError:
        print("(bpe.py not found, skipping comparison)")

    # ── Block 7: Real corpus test ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Block 7: WordPiece trained on data/test_bpe.txt")
    print("=" * 60)

    import os
    corpus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_bpe.txt")
    with open(corpus_path, encoding="utf-8") as _f:
        real_corpus = [line.strip() for line in _f if line.strip()]

    target_vocab = 500
    print(f"Corpus: {len(real_corpus)} lines  |  Target vocab: {target_vocab}")

    wp_real = WordPieceTokenizer()
    wp_real.train(real_corpus, vocab_size=target_vocab, fast=True)
    print(f"Vocab size: {len(wp_real.vocab)}  |  Merges learned: {len(wp_real.merges)}")

    long_toks = sorted(wp_real.vocab, key=len, reverse=True)[:15]
    print(f"Longest tokens (top 15): {long_toks}")

    print("\nSample tokenizations:")
    test_words = ["harpooneer", "landlord", "sleeping", "whale", "cannibal", "unknown", "whaling"]
    for w in test_words:
        print(f"  {w!r:>14} -> {wp_real.tokenize(w)}")
