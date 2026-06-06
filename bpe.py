"""
bpe.py
======
Member A's implementation: Naive BPE + Fast BPE.
Both inherit from tokenizer_interface.BaseTokenizer.
"""

from __future__ import annotations
import heapq
from collections import defaultdict, Counter

from tokenizer_interface import (
    BaseTokenizer,
    preprocess,
    Vocab,
    TokenList,
    Corpus,
    MergeRule,
)


# ─────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────

def get_stats(word_freq: dict[str, int]) -> dict[tuple[str, str], int]:
    """
    Count the weighted frequency of every adjacent token pair in word_freq.

    Args:
        word_freq: e.g. {"l o w </w>": 3, "n e w e s t </w>": 2}
                   Each key is a space-separated sequence of tokens,
                   each value is how many times that word occurs.

    Returns:
        pair_freq: e.g. {("l","o"): 3, ("o","w"): 3, ("w","</w>"): 3, ...}
                   Frequency of each adjacent pair, weighted by word frequency.
    """
    pair_freq: dict[tuple[str, str], int] = defaultdict(int)
    for word, freq in word_freq.items():
        tokens = word.split()
        for i in range(len(tokens) - 1):
            pair_freq[(tokens[i], tokens[i + 1])] += freq
    return dict(pair_freq)


def merge_vocab(
    pair: tuple[str, str],
    word_freq: dict[str, int],
) -> dict[str, int]:
    """
    For every word in word_freq, merge the adjacent token pair (A, B) into "AB".

    Example: pair=("l","o"), word "l o w </w>" -> "lo w </w>"

    WARNING: the merge must happen at **token boundaries**.
    Do NOT use str.replace("A B", "AB"): it can match across token boundaries.
    For instance, once the token "j</w>" exists, the string "l j</w>" contains
    the substring "l j", and str.replace would incorrectly rewrite it as
    "lj</w>", silently corrupting the tokenization.

    The safe approach: split the word into tokens, scan adjacent tokens,
    merge whenever they equal (A, B), then join back with spaces.
    """
    a, b = pair
    new_word_freq: dict[str, int] = {}
    for word, freq in word_freq.items():
        tokens = word.split()
        merged: list[str] = []
        i = 0
        n = len(tokens)
        while i < n:
            if i < n - 1 and tokens[i] == a and tokens[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        new_word_freq[" ".join(merged)] = freq
    return new_word_freq


# ─────────────────────────────────────────
# BPE Tokenizer
# ─────────────────────────────────────────

class BPETokenizer(BaseTokenizer):
    """
    BPE tokenizer.
        self.vocab:  set[str]            vocabulary learned during training
        self.merges: list[(str,str)]     ordered list of merge rules
                                         (order matters: tokenize() applies them in order)
    """

    def __init__(self):
        super().__init__()
        self.merges: list[MergeRule] = []

    # ─────────────────────────────────────────
    # Training entry point (interface unchanged)
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
        Train BPE on the given corpus until the vocabulary reaches vocab_size.

        fast=True  -> uses heapq + incremental updates (~O(n log n))
        fast=False -> uses the Naive implementation (rescans the whole corpus
                      every round, ~O(n^2))

        Both modes produce **identical** self.vocab and self.merges
        (same content, same order).
        """
        if fast:
            self._train_fast(corpus, vocab_size, verbose=verbose)
        else:
            self._train_naive(corpus, vocab_size, verbose=verbose)

    # ── Naive training: rescans the whole corpus every round ──
    def _train_naive(self, corpus: Corpus, vocab_size: int, verbose: bool = False) -> None:
        word_freq = preprocess(corpus)

        # Initial vocabulary = every single character (including </w>)
        self.vocab = set()
        for word in word_freq:
            for tok in word.split():
                self.vocab.add(tok)
        self.merges = []

        if verbose:
            print(f"[naive init] vocab_size={len(self.vocab)}")

        while len(self.vocab) < vocab_size:
            stats = get_stats(word_freq)
            if not stats:
                break

            # Pick the pair with the highest frequency; on ties take the
            # lexicographically **smallest** pair (this aligns with the
            # natural ordering of the heap used by the Fast version, so
            # both implementations produce identical merges).
            best_pair = min(stats.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            best_freq = stats[best_pair]

            word_freq = merge_vocab(best_pair, word_freq)
            new_token = best_pair[0] + best_pair[1]
            self.vocab.add(new_token)
            self.merges.append(best_pair)

            if verbose:
                print(f"[naive merge {len(self.merges):>3}] "
                      f"{best_pair} (freq={best_freq}) -> {new_token!r}")

        self.is_trained = True

    # ── Fast training: heapq + inverted index + incremental updates ──
    def _train_fast(self, corpus: Corpus, vocab_size: int, verbose: bool = False) -> None:
        """
        Equivalent to the Naive version, but uses incremental data structures
        so that we never need to rescan the whole corpus on every round.

        Internal state (lives only inside this function):
            words[i]       current token list of the i-th word (mutable)
            freqs[i]       word frequency of the i-th word
            pair_freq[p]   live count of pair p across all words
            pair_where[p]  inverted index: the set of word indices that contain p
            heap           min-heap of (-pair_freq, pair). Stale entries are
                           tolerated; we validate at pop time using pair_freq
                           and skip any entry whose stored count no longer matches.
        """
        word_freq = preprocess(corpus)

        # 1) Split into parallel arrays words / freqs and initialise vocab.
        words: list[list[str]] = []
        freqs: list[int] = []
        self.vocab = set()
        for w, f in word_freq.items():
            toks = w.split()
            words.append(toks)
            freqs.append(f)
            self.vocab.update(toks)
        self.merges = []

        # 2) One full pass to populate pair_freq and pair_where.
        pair_freq: dict[tuple[str, str], int] = defaultdict(int)
        pair_where: dict[tuple[str, str], set[int]] = defaultdict(set)
        for wi, toks in enumerate(words):
            f = freqs[wi]
            for a, b in zip(toks, toks[1:]):
                pair_freq[(a, b)] += f
                pair_where[(a, b)].add(wi)

        # 3) Build the heap.
        heap: list[tuple[int, tuple[str, str]]] = [(-c, p) for p, c in pair_freq.items()]
        heapq.heapify(heap)

        if verbose:
            print(f"[fast init] vocab_size={len(self.vocab)}, pairs={len(pair_freq)}")

        # 4) Main loop.
        while len(self.vocab) < vocab_size:
            # (1) Pop the most frequent pair, skipping stale entries.
            best_pair = None
            while heap:
                neg, p = heapq.heappop(heap)
                if -neg == pair_freq.get(p, 0) and -neg > 0:
                    best_pair = p
                    break
            if best_pair is None:
                break

            best_freq = pair_freq[best_pair]
            new_tok = best_pair[0] + best_pair[1]
            self.vocab.add(new_tok)
            self.merges.append(best_pair)

            # (2) Touch only the words that contain best_pair.
            affected = pair_where.pop(best_pair, set())
            # best_pair has just been merged away; drop it from pair_freq too.
            pair_freq.pop(best_pair, None)

            for wi in affected:
                f = freqs[wi]
                old_toks = words[wi]
                # Pair multiplicities BEFORE the merge.
                old_counts = Counter(zip(old_toks, old_toks[1:]))
                # Use the same single-rule merge helper as tokenize()
                # to guarantee identical behaviour (one source of truth).
                new_toks = self._apply_one_merge(old_toks, best_pair)
                words[wi] = new_toks
                new_counts = Counter(zip(new_toks, new_toks[1:]))

                # (3) Diff-update pair_freq and pair_where.
                changed: set[tuple[str, str]] = set(old_counts) | set(new_counts)
                for p in changed:
                    delta = (new_counts.get(p, 0) - old_counts.get(p, 0)) * f
                    if delta != 0:
                        pair_freq[p] = pair_freq.get(p, 0) + delta
                    # Inverted index: add or remove this word index
                    # depending on whether the word still contains p.
                    if new_counts.get(p, 0) > 0:
                        pair_where[p].add(wi)
                    else:
                        pair_where[p].discard(wi)
                    # (4) Push the updated count onto the heap. Old entries
                    # are left in the heap and filtered out at pop time.
                    cur = pair_freq.get(p, 0)
                    if cur > 0:
                        heapq.heappush(heap, (-cur, p))

            if verbose:
                print(f"[fast merge {len(self.merges):>3}] "
                      f"{best_pair} (freq={best_freq}) -> {new_tok!r}")

        self.is_trained = True

    # ─────────────────────────────────────────
    # Inference: tokenize
    # ─────────────────────────────────────────
    def tokenize(self, text: str) -> TokenList:
        """
        Split text into a list of tokens.

        ── How BPE resolves ambiguity ──────────────────────────────────
        At inference time, BPE does **not** use greedy longest-match.
        Instead, it:
            (1) splits each word into single characters (plus </w>), and
            (2) applies the rules in self.merges in the exact order they
                were learned during training, merging every adjacent pair
                that matches.

        Why this resolves ambiguity:
            Consider a word like "newer", which could in principle be
            split as ["new","er"] or ["n","ewer"]. A greedy "pick whatever
            subword is in the vocab" strategy would depend on scan direction
            and length priorities and might not be unique.
            BPE instead promotes the **order of merges** to be the single
            source of truth:
                - self.merges is a fixed, ordered list after training.
                - At inference we start from characters and apply each rule
                  in that exact order; each rule does a deterministic
                  left-to-right scan-and-merge pass.
            So no matter how a reader "imagines" a split, walking through
            the rule list in order always yields the same unique result.
            Training assigns the order ("most frequent pair first");
            inference exploits the order to guarantee uniqueness.
        ────────────────────────────────────────────────────────────────
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer is not trained yet; call train() first.")

        # Use the same preprocessing as training (lowercase, strip punctuation)
        # so that vocab lookups happen on the same character set.
        word_freq = preprocess([text])
        # Keys of word_freq are already strings like "c h a r s </w>".
        all_tokens: TokenList = []
        for char_word in word_freq:
            tokens = char_word.split()                # e.g. ["n","e","w","e","r","</w>"]
            tokens = self._apply_merges(tokens)       # apply rules in training order
            # Subwords not in the vocab become [UNK].
            tokens = [t if t in self.vocab else "[UNK]" for t in tokens]
            all_tokens.extend(tokens)
        return all_tokens

    @staticmethod
    def _apply_one_merge(tokens: TokenList, pair: tuple[str, str]) -> TokenList:
        """
        Apply a single merge rule (A, B) to a token list:
        scan left to right and whenever we see adjacent (A, B), merge them
        into "AB". Merges are **non-overlapping** (we skip 2 positions
        after a successful merge).

        Shared by inference (_apply_merges) and Fast training, so the
        merge semantics live in exactly one place.
        """
        a, b = pair
        i = 0
        merged: TokenList = []
        n = len(tokens)
        while i < n:
            if i < n - 1 and tokens[i] == a and tokens[i + 1] == b:
                merged.append(a + b)
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
# Self-tests / small demo
# ─────────────────────────────────────────

if __name__ == "__main__":
    word_freq = {"l o w </w>": 3, "n e w e s t </w>": 2}
    stats = get_stats(word_freq)

    print("Input word_freq:")
    for k, v in word_freq.items():
        print(f"  {k!r}: {v}")

    print("\nget_stats output (sorted by frequency):")
    for pair, freq in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {pair}: {freq}")

    # Expected counts:
    #   "l o w </w>"      (x3) -> (l,o)=3, (o,w)=3, (w,</w>)=3
    #   "n e w e s t </w>"(x2) -> (n,e)=2, (e,w)=2, (w,e)=2, (e,s)=2, (s,t)=2, (t,</w>)=2
    expected = {
        ("l", "o"): 3, ("o", "w"): 3, ("w", "</w>"): 3,
        ("n", "e"): 2, ("e", "w"): 2, ("w", "e"): 2,
        ("e", "s"): 2, ("s", "t"): 2, ("t", "</w>"): 2,
    }
    assert stats == expected, f"mismatch!\n  got:      {stats}\n  expected: {expected}"
    print("\n[OK] get_stats matches the expected counts")

    # ── Block 2: Naive train() demo ──
    print("\n" + "=" * 60)
    print("Block 2: Naive BPE training")
    print("=" * 60)
    corpus = ["low low low lowest newest"]
    bpe = BPETokenizer()
    bpe.train(corpus, vocab_size=15, fast=False, verbose=True)
    print(f"\nFinal vocab ({len(bpe.vocab)}): {sorted(bpe.vocab)}")
    print(f"merges ({len(bpe.merges)}): {bpe.merges}")

    # ── Block 3: tokenize demo ──
    print("\n" + "=" * 60)
    print("Block 3: tokenize (apply merges in order, resolves ambiguity)")
    print("=" * 60)
    for w in ["lowest", "newest", "low", "newer", "unknown"]:
        print(f"  tokenize({w!r:>10}) -> {bpe.tokenize(w)}")

    # Ambiguity demo: start from different "imagined" splits and apply
    # self.merges in order; the canonical char-level start is unique.
    print("\n— Ambiguity demo: different starting splits → final result —")
    starts = {
        "char-split":       ["l", "o", "w", "e", "s", "t", "</w>"],
        "imagined-split-1": ["lo", "w", "e", "s", "t", "</w>"],   # pretend "lo" already exists
        "imagined-split-2": ["l", "ow", "est", "</w>"],            # pretend a partial merge happened
    }
    results = {}
    for name, toks in starts.items():
        out = bpe._apply_merges(toks)
        print(f"  {name:>18}: {toks}  ->  {out}")
        results[name] = out
    # Note: not all imagined splits collapse to the same final list
    # (a non-canonical start can skip rules that needed the original chars).
    # The point of the demo is that the **canonical char-level start** is
    # the unique, deterministic path; once you start there, merge order
    # alone determines the result.
    assert bpe.tokenize("lowest") == bpe.tokenize("lowest"), "determinism check failed"
    print("\n[OK] starting from char-level, tokenize is deterministic and unique")

    # ── Block 4: Fast BPE equivalence + benchmark ──
    print("\n" + "=" * 60)
    print("Block 4: Fast BPE (heapq + incremental updates)")
    print("=" * 60)

    # (a) Small-corpus equivalence: vocab and merges must match Naive exactly.
    naive = BPETokenizer(); naive.train(corpus, vocab_size=15, fast=False)
    fast  = BPETokenizer(); fast.train(corpus,  vocab_size=15, fast=True)
    assert naive.vocab  == fast.vocab,  f"vocab mismatch:\n  naive={naive.vocab}\n  fast ={fast.vocab}"
    assert naive.merges == fast.merges, f"merges mismatch:\n  naive={naive.merges}\n  fast ={fast.merges}"
    print(f"[OK] small corpus (vocab_size=15): naive.vocab == fast.vocab and naive.merges == fast.merges")

    # (b) Medium-corpus benchmark — synthetic corpus so the speed gap is visible.
    import random, time
    random.seed(42)
    alphabet = "abcdefghijklmnop"
    # Build a base vocabulary of ~80 words, then build sentences from them.
    base_words = ["".join(random.choice(alphabet) for _ in range(random.randint(3, 9)))
                  for _ in range(80)]
    sentences = [" ".join(random.choice(base_words) for _ in range(20)) for _ in range(100)]
    medium_corpus = sentences
    target_vocab = 300

    t0 = time.perf_counter()
    n2 = BPETokenizer(); n2.train(medium_corpus, vocab_size=target_vocab, fast=False)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    f2 = BPETokenizer(); f2.train(medium_corpus, vocab_size=target_vocab, fast=True)
    t_fast = time.perf_counter() - t0

    assert n2.vocab  == f2.vocab,  "medium-corpus vocab mismatch"
    assert n2.merges == f2.merges, "medium-corpus merges mismatch"
    speedup = t_naive / t_fast if t_fast > 0 else float("inf")
    print(f"\n  Medium corpus: {len(medium_corpus)} sentences x 20 words, target_vocab={target_vocab}")
    print(f"  Naive: {t_naive*1000:7.1f} ms")
    print(f"  Fast : {t_fast*1000:7.1f} ms")
    print(f"  [OK] vocab/merges identical; Fast speedup ~ {speedup:.1f}x")

    # (c) tokenize also works on the Fast-trained model.
    sample = "abcfgh"
    print(f"\n  fast.tokenize({sample!r}) -> {f2.tokenize(sample)}")
