"""
metrics.py
==========
Evaluation metric library, shared by run_eval.py.

Two gold-standard sources, with a shared query interface
(`db.segmentation: dict[word -> list[morpheme]]`, `db.get_morphemes(word)`),
so boundary_prf / morpheme_recovery_rate / over_under_segmentation accept
either one (duck typing), while affix_detection / segmentation_consistency
require MorphoLexDB specifically (they need `.roles`):

  - MorphoLexDB: canonical-form morphemes + prefix/root/suffix roles.
    Not always character-aligned with the actual spelling.
  - GoldStdDB: surface-form morphemes, 100% character-aligned, no roles.

Metric groups:

  Reference-free (no gold standard needed, run over a test corpus):
    WordFreqProfile, TokenizationCache (walks every word occurrence in
    the corpus, memoizing repeats so total_sec reflects processing the
    whole text without redundant retokenizing), fertility_stats,
    wordtype_stats, token_length_dist, unk_rate, intact_token_stats,
    basic_stats.

  Reference-based (need a gold standard, run over a word list):
    boundary_prf, morpheme_recovery_rate, affix_detection,
    segmentation_consistency, over_under_segmentation.

This module only computes; it does not print progress or write files.
Callers (run_eval.py) own all user-facing output and persistence.
"""

from __future__ import annotations
import time, re
from dataclasses import dataclass, field
from collections import defaultdict
from typing import TYPE_CHECKING, Union

from tokenizer_interface import BaseTokenizer

if TYPE_CHECKING:
    from load_data import MorphoLexDB, GoldStdDB

# Anything with a `.segmentation` dict can be passed to the boundary-style metrics.
SegmentationSource = Union["MorphoLexDB", "GoldStdDB"]


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def clean_token(tok: str) -> str:
    """Strip WordPiece's `##` prefix and BPE's `</w>` end marker, lowercase."""
    return tok.replace("##", "").replace("</w>", "").lower()


def tokens_to_boundaries(tokens: list[str]) -> set[int]:
    """Token list -> set of character split positions. ["play","##ing"] -> {4}"""
    cleaned = [clean_token(t) for t in tokens]
    boundaries, pos = set(), 0
    for seg in cleaned[:-1]:
        pos += len(seg)
        boundaries.add(pos)
    return boundaries


def morphemes_to_boundaries(morphemes: list[str]) -> set[int]:
    """Gold morpheme list -> set of character split positions."""
    boundaries, pos = set(), 0
    for m in morphemes[:-1]:
        pos += len(m)
        boundaries.add(pos)
    return boundaries


# ═════════════════════════════════════════════════════════════════════════════
# WordFreqProfile — word-frequency counts from a test corpus
# ═════════════════════════════════════════════════════════════════════════════

def extract_words(corpus: list[str], lowercase: bool = True) -> list[str]:
    """Every word occurrence in the corpus, in original order (repeats included)."""
    words: list[str] = []
    for sentence in corpus:
        s = sentence.lower() if lowercase else sentence
        words.extend(re.findall(r"[a-zA-Z]+", s))
    return words


@dataclass
class WordFreqProfile:
    """
    Word-frequency counts over a test corpus; the basis for all
    high-frequency-vs-low-frequency comparisons.

    high_freq / low_freq each take a fixed count (default 1000), not a
    percentage, so the group sizes stay comparable across corpora of
    different sizes.
    """
    word_counts:  dict[str, int]
    all_words:    list[str]     # unique word types, sorted by frequency desc
    high_freq:    list[str]
    low_freq:     list[str]
    total_tokens: int

    @classmethod
    def from_corpus(cls, corpus: list[str], top_n: int = 1000,
                    lowercase: bool = True) -> "WordFreqProfile":
        counts: dict[str, int] = defaultdict(int)
        for w in extract_words(corpus, lowercase=lowercase):
            counts[w] += 1

        sorted_words = sorted(counts.keys(), key=lambda w: -counts[w])
        high_freq = sorted_words[:top_n]
        low_freq  = sorted_words[-top_n:] if len(sorted_words) >= top_n else sorted_words

        return cls(
            word_counts=dict(counts),
            all_words=sorted_words,
            high_freq=high_freq,
            low_freq=low_freq,
            total_tokens=sum(counts.values()),
        )

    def summary(self) -> dict:
        return {
            "n_wordtypes":  len(self.all_words),
            "n_tokens":     self.total_tokens,
            "n_high_freq":  len(self.high_freq),
            "n_low_freq":   len(self.low_freq),
            "high_freq_min_count": self.word_counts[self.high_freq[-1]] if self.high_freq else 0,
            "low_freq_max_count":  self.word_counts[self.low_freq[-1]]  if self.low_freq  else 0,
        }


# ═════════════════════════════════════════════════════════════════════════════
# TokenizationCache — tokenize each unique word type once, cache word -> tokens
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TokenizationCache:
    """
    Tokenizes a word-occurrence stream (e.g. every word in a test corpus,
    in original order, repeats included) and stores word -> token list,
    so the same (tokenizer, word stream) can be saved to disk and
    replayed without re-tokenizing.

    Tokenizing is by far the most expensive step in evaluation (BPE
    re-applies every learned merge rule to every word), so a repeated
    word is looked up instead of re-tokenized (memoized in `results`).
    But the timed walk still covers every occurrence in the stream, not
    just the unique words: a word always tokenizes the same way
    regardless of context, so memoizing repeats changes nothing about
    correctness, while `total_sec` still reflects the cost of processing
    the whole text (dominated by the unique-word tokenize() calls, plus
    a cheap dict lookup per repeat) rather than just its vocabulary size.

    Attributes:
        results   : dict[word -> tokens]
        cleaned   : dict[word -> tokens]   (`##`/`</w>` stripped; derived)
        total_sec : float   time spent walking the word stream
    """
    results:   dict[str, list[str]]
    total_sec: float = 0.0
    cleaned:   dict[str, list[str]] = field(init=False, repr=False)

    def __post_init__(self):
        self.cleaned = {w: [clean_token(t) for t in toks]
                        for w, toks in self.results.items()}

    @classmethod
    def build(cls, tokenizer: BaseTokenizer, words: list[str]) -> "TokenizationCache":
        """
        Walk `words` in order (repeats allowed); a word already seen is
        looked up in `results` instead of re-tokenized. Times the whole
        walk, so total_sec approximates tokenizing the full stream.
        """
        results: dict[str, list[str]] = {}
        t0 = time.perf_counter()
        for w in words:
            if w not in results:
                results[w] = tokenizer.tokenize(w)
        elapsed = time.perf_counter() - t0
        return cls(results=results, total_sec=round(elapsed, 4))

    def n_tokens(self, word: str) -> int:
        """How many tokens a word was split into (cache lookup, no re-tokenizing)."""
        return len(self.results.get(word, []))

    def is_intact(self, word: str) -> bool:
        """Whether the word stayed a single token (and isn't [UNK])."""
        toks = self.results.get(word, [])
        return len(toks) == 1 and toks[0] != "[UNK]"

    def has_unk(self, word: str) -> bool:
        return "[UNK]" in self.results.get(word, [])

    def token_lengths(self, word: str) -> list[int]:
        """Character length of each (cleaned) token for this word."""
        return [len(t) for t in self.cleaned.get(word, [])]

    def lookup(self, word: str) -> list[str]:
        """Cached tokens for a single word (empty list if not in the cache)."""
        return self.results.get(word.lower(), [])

    def save_to_dict(self) -> dict:
        return {"results": self.results, "total_sec": self.total_sec}

    @classmethod
    def load_from_dict(cls, data: dict) -> "TokenizationCache":
        """Rebuild a cache from a previously saved JSON dict (no tokenizer needed)."""
        return cls(results=data["results"], total_sec=data.get("total_sec", 0.0))


class CachedTokenizer:
    """
    Adapts a TokenizationCache to look like a tokenizer (`.tokenize()`,
    `.vocab_size()`), so the reference-based metric functions below
    (which only call those two methods on single words) can run against
    cached results without re-tokenizing.
    """

    def __init__(self, cache: TokenizationCache, vocab_size: int):
        self._cache = cache
        self._vocab_size = vocab_size

    def tokenize(self, text: str) -> list[str]:
        return self._cache.lookup(text)

    def vocab_size(self) -> int:
        return self._vocab_size


# ═════════════════════════════════════════════════════════════════════════════
# Fertility — average tokens per word occurrence (frequency-weighted)
# ═════════════════════════════════════════════════════════════════════════════

def fertility_stats(cache: TokenizationCache, profile: WordFreqProfile) -> dict:
    """fertility = sum(n_tokens(w) * count(w)) / sum(count(w)), over all word types."""
    if not profile.all_words:
        return {"fertility": 0.0, "n_word_occurrences": 0}

    total_subwords, total_occurrences = 0, 0
    for w in profile.all_words:
        n_subwords = cache.n_tokens(w)
        count      = profile.word_counts[w]
        total_subwords    += n_subwords * count
        total_occurrences += count

    return {
        "fertility":          round(total_subwords / total_occurrences, 4),
        "n_word_occurrences": total_occurrences,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Average tokens per wordtype (unweighted) + high/low-frequency breakdown
# ═════════════════════════════════════════════════════════════════════════════

def _avg_tokens_per_wordtype(cache: TokenizationCache, words: list[str]) -> float:
    if not words:
        return 0.0
    return sum(cache.n_tokens(w) for w in words) / len(words)


def wordtype_stats(cache: TokenizationCache, profile: WordFreqProfile) -> dict:
    """Average tokens per unique word type, with all/high_freq/low_freq breakdowns."""
    return {
        "all": {
            "avg_tokens":  round(_avg_tokens_per_wordtype(cache, profile.all_words), 4),
            "n_wordtypes": len(profile.all_words),
        },
        "high_freq": {
            "avg_tokens":  round(_avg_tokens_per_wordtype(cache, profile.high_freq), 4),
            "n_wordtypes": len(profile.high_freq),
        },
        "low_freq": {
            "avg_tokens":  round(_avg_tokens_per_wordtype(cache, profile.low_freq), 4),
            "n_wordtypes": len(profile.low_freq),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Token length distribution
# ═════════════════════════════════════════════════════════════════════════════

def token_length_dist(cache: TokenizationCache, words: list[str]) -> dict[int, int]:
    """{token character length -> occurrence count}, from cache."""
    counter: dict[int, int] = defaultdict(int)
    for w in words:
        for length in cache.token_lengths(w):
            counter[length] += 1
    return dict(sorted(counter.items()))


# ═════════════════════════════════════════════════════════════════════════════
# UNK rate
# ═════════════════════════════════════════════════════════════════════════════

def unk_rate(cache: TokenizationCache, words: list[str]) -> float:
    """Fraction of words whose tokenization contains [UNK]."""
    if not words:
        return 0.0
    return sum(1 for w in words if cache.has_unk(w)) / len(words)


# ═════════════════════════════════════════════════════════════════════════════
# Intact token rate — how many words stay a single whole token
# ═════════════════════════════════════════════════════════════════════════════

def intact_token_stats(cache: TokenizationCache, profile: WordFreqProfile) -> dict:
    """All/high_freq/low_freq breakdown of intact (unsplit) word rate."""
    def _calc(words: list[str]) -> dict:
        if not words:
            return {"intact_count": 0, "total": 0, "intact_pct": 0.0}
        intact_count = sum(1 for w in words if cache.is_intact(w))
        return {
            "intact_count": intact_count,
            "total":        len(words),
            "intact_pct":   round(intact_count / len(words) * 100, 2),
        }

    return {
        "all":       _calc(profile.all_words),
        "high_freq": _calc(profile.high_freq),
        "low_freq":  _calc(profile.low_freq),
    }


# ═════════════════════════════════════════════════════════════════════════════
# basic_stats — run all reference-free metrics over one (tokenizer, corpus) pair
# ═════════════════════════════════════════════════════════════════════════════

def basic_stats(tokenizer: BaseTokenizer, corpus: list[str],
                top_n: int = 1000,
                cache: TokenizationCache | None = None) -> dict:
    """
    Run the full reference-free metric suite for one tokenizer on one
    test corpus. Tokenizes each unique word type once (unless `cache` is
    already supplied, e.g. loaded from disk), then derives every metric
    from that single cache.

    Returns a flat dict (plus "_cache", the TokenizationCache instance,
    for the caller to persist via `cache.save_to_dict()`).
    """
    profile = WordFreqProfile.from_corpus(corpus, top_n=top_n)

    if cache is None:
        cache = TokenizationCache.build(tokenizer, extract_words(corpus))

    return {
        "vocab_size":         tokenizer.vocab_size(),
        "total_sec":          cache.total_sec,
        "fertility":          fertility_stats(cache, profile),
        "wordtype_stats":     wordtype_stats(cache, profile),
        "intact_token_stats": intact_token_stats(cache, profile),
        "unk_rate":           round(unk_rate(cache, profile.all_words), 4),
        "token_length_dist":  token_length_dist(cache, profile.all_words),
        "profile_summary":    profile.summary(),
        "_cache":             cache,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Vocabulary overlap (cross-algorithm)
# ═════════════════════════════════════════════════════════════════════════════

def vocab_overlap(tok_a: BaseTokenizer, tok_b: BaseTokenizer,
                  name_a: str = "A", name_b: str = "B") -> dict:
    """Jaccard overlap between two tokenizers' vocabularies (cleaned tokens)."""
    va = {clean_token(t) for t in tok_a.vocab if clean_token(t)}
    vb = {clean_token(t) for t in tok_b.vocab if clean_token(t)}
    shared = va & vb
    union  = va | vb
    return {
        "jaccard":          round(len(shared) / len(union), 4) if union else 0.0,
        "shared_count":     len(shared),
        f"only_{name_a}_count": len(va - vb),
        f"only_{name_b}_count": len(vb - va),
        f"only_{name_a}_examples": sorted(va - vb)[:20],
        f"only_{name_b}_examples": sorted(vb - va)[:20],
        "shared_examples":  sorted(shared)[:20],
    }


# ═════════════════════════════════════════════════════════════════════════════
# Boundary Precision / Recall / F1
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class WordBoundaryScore:
    word:      str
    predicted: list[str]
    gold:      list[str]
    P: float = 0.0
    R: float = 0.0
    F1: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0


def _score_one_word(word: str, predicted: list[str], gold: list[str]) -> WordBoundaryScore:
    pred_b = tokens_to_boundaries(predicted)
    gold_b = morphemes_to_boundaries(gold)
    tp = len(pred_b & gold_b)
    fp = len(pred_b - gold_b)
    fn = len(gold_b - pred_b)
    P  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = 2*P*R / (P+R)  if (P + R)  > 0 else 0.0
    return WordBoundaryScore(word=word, predicted=predicted, gold=gold,
                              P=P, R=R, F1=F1, tp=tp, fp=fp, fn=fn)


@dataclass
class BoundaryEvalResult:
    """Boundary P/R/F1 aggregated over a test-word set."""
    algorithm:  str
    vocab_size: int
    n_words:    int
    macro_P:    float
    macro_R:    float
    macro_F1:   float
    micro_P:    float
    micro_R:    float
    micro_F1:   float
    word_scores: list[WordBoundaryScore] = field(default_factory=list, repr=False)

    def worst_n(self, n: int = 10) -> list[WordBoundaryScore]:
        return sorted(self.word_scores, key=lambda s: s.F1)[:n]

    def best_n(self, n: int = 10) -> list[WordBoundaryScore]:
        return sorted(self.word_scores, key=lambda s: s.F1, reverse=True)[:n]


def boundary_prf(tokenizer: BaseTokenizer, algorithm: str,
                 test_words: list[str], db: SegmentationSource) -> BoundaryEvalResult:
    """
    Character-level boundary P/R/F1 over test_words. Use a GoldStdDB
    (surface-aligned), not MorphoLexDB (canonical form may not align
    with the word's actual spelling, which would corrupt boundary
    positions). Only words with >=2 gold morphemes are scored (a
    single-morpheme word has no boundary to predict).
    """
    scores = []
    eval_words = [w for w in test_words
                  if w in db.segmentation and len(db.segmentation[w]) >= 2]

    for word in eval_words:
        tokens = tokenizer.tokenize(word)
        gold   = db.segmentation[word]
        scores.append(_score_one_word(word, tokens, gold))

    macro_P  = sum(s.P  for s in scores) / len(scores) if scores else 0.0
    macro_R  = sum(s.R  for s in scores) / len(scores) if scores else 0.0
    macro_F1 = sum(s.F1 for s in scores) / len(scores) if scores else 0.0
    ttp = sum(s.tp for s in scores)
    tfp = sum(s.fp for s in scores)
    tfn = sum(s.fn for s in scores)
    micro_P  = ttp / (ttp + tfp) if (ttp + tfp) > 0 else 0.0
    micro_R  = ttp / (ttp + tfn) if (ttp + tfn) > 0 else 0.0
    micro_F1 = 2*micro_P*micro_R / (micro_P+micro_R) if (micro_P+micro_R) > 0 else 0.0

    return BoundaryEvalResult(
        algorithm=algorithm, vocab_size=tokenizer.vocab_size(),
        n_words=len(scores),
        macro_P=macro_P, macro_R=macro_R, macro_F1=macro_F1,
        micro_P=micro_P, micro_R=micro_R, micro_F1=micro_F1,
        word_scores=scores,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Morpheme Recovery Rate
# ═════════════════════════════════════════════════════════════════════════════

def morpheme_recovery_rate(tokenizer: BaseTokenizer, test_words: list[str],
                           db: SegmentationSource) -> float:
    """
    Per word: |cleaned predicted tokens ∩ gold morphemes| / |gold morphemes|,
    macro-averaged. Prefer a GoldStdDB — string equality against a
    canonical-form gold (MorphoLexDB) would unfairly penalize correct
    splits that don't match the canonical spelling.
    """
    rates = []
    for word in test_words:
        gold = db.segmentation.get(word)
        if not gold:
            continue
        pred_clean = {clean_token(t) for t in tokenizer.tokenize(word)}
        gold_set   = set(gold)
        rates.append(len(pred_clean & gold_set) / len(gold_set))
    return round(sum(rates) / len(rates), 4) if rates else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Affix Detection (Prefix Recall + Suffix Recall) — requires MorphoLexDB
# ═════════════════════════════════════════════════════════════════════════════

def affix_detection(tokenizer: BaseTokenizer, test_words: list[str],
                    db: "MorphoLexDB", n_examples: int = 15) -> dict:
    """
    Recall of gold prefixes/suffixes among the tokenizer's (cleaned)
    output tokens. Requires MorphoLexDB for `.roles`. Only recall is
    computed (the tokenizer doesn't label which of its tokens are
    "supposed" to be prefixes, so precision isn't well-defined here).
    """
    prefix_hits, prefix_total = 0, 0
    suffix_hits, suffix_total = 0, 0
    prefix_examples, suffix_examples = [], []

    for word in test_words:
        gold_morphemes = db.segmentation.get(word)
        gold_roles     = db.roles.get(word)
        if not gold_morphemes or not gold_roles:
            continue

        pred_clean = [clean_token(t) for t in tokenizer.tokenize(word)]
        pred_set   = set(pred_clean)

        for m, role in zip(gold_morphemes, gold_roles):
            detected = m in pred_set
            if role == "prefix":
                prefix_total += 1
                prefix_hits += int(detected)
                if len(prefix_examples) < n_examples:
                    prefix_examples.append({"word": word, "gold_prefix": m,
                                            "detected": detected, "tokens": pred_clean})
            elif role == "suffix":
                suffix_total += 1
                suffix_hits += int(detected)
                if len(suffix_examples) < n_examples:
                    suffix_examples.append({"word": word, "gold_suffix": m,
                                            "detected": detected, "tokens": pred_clean})

    return {
        "prefix_recall":   round(prefix_hits / prefix_total, 4) if prefix_total > 0 else 0.0,
        "suffix_recall":   round(suffix_hits / suffix_total, 4) if suffix_total > 0 else 0.0,
        "prefix_total":    prefix_total,
        "suffix_total":    suffix_total,
        "prefix_examples": prefix_examples,
        "suffix_examples": suffix_examples,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Segmentation Consistency — requires MorphoLexDB
# ═════════════════════════════════════════════════════════════════════════════

def segmentation_consistency(tokenizer: BaseTokenizer, test_words: list[str],
                              db: "MorphoLexDB", min_appearances: int = 3,
                              n_roots: int = 15) -> dict:
    """
    For roots that appear in >= min_appearances test words, how often
    is the root consistently sliced out across all of them? Requires
    MorphoLexDB for `.roles` (root identification).
    """
    root_to_words: dict[str, list[str]] = defaultdict(list)
    for word in test_words:
        morphemes = db.segmentation.get(word, [])
        roles     = db.roles.get(word, [])
        for m, r in zip(morphemes, roles):
            if r == "root":
                root_to_words[m].append(word)

    eligible = {r: ws for r, ws in root_to_words.items() if len(ws) >= min_appearances}

    root_details = []
    for root, words_with_root in sorted(eligible.items(), key=lambda x: -len(x[1])):
        breakdown = []
        detected_count = 0
        for word in words_with_root:
            pred_clean = [clean_token(t) for t in tokenizer.tokenize(word)]
            detected   = root in pred_clean
            detected_count += int(detected)
            breakdown.append({"word": word, "tokens": pred_clean, "detected": detected})
        consistency = detected_count / len(words_with_root)
        root_details.append({
            "root":           root,
            "appearances":    len(words_with_root),
            "detected_count": detected_count,
            "consistency":    round(consistency, 4),
            "word_breakdown": breakdown,
        })
        if len(root_details) >= n_roots:
            break

    mean_c = (sum(d["consistency"] for d in root_details) / len(root_details)
              if root_details else 0.0)
    return {
        "mean_consistency":  round(mean_c, 4),
        "n_roots_evaluated": len(root_details),
        "root_details":      root_details,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Over-segmentation / Under-segmentation
# ═════════════════════════════════════════════════════════════════════════════

def over_under_segmentation(tokenizer: BaseTokenizer, test_words: list[str],
                            db: SegmentationSource, n_examples: int = 15) -> dict:
    """
    Classifies each word by token-count vs. gold-morpheme-count:
    over-segmentation (more tokens than morphemes), under-segmentation
    (fewer), or perfect (equal count — not necessarily aligned
    boundaries; see boundary_prf for that).
    """
    over_seg, under_seg, perfect = [], [], []

    for word in test_words:
        gold = db.segmentation.get(word)
        if not gold:
            continue
        pred = [clean_token(t) for t in tokenizer.tokenize(word) if clean_token(t)]
        entry = {"word": word, "pred": pred, "gold": gold,
                 "pred_n": len(pred), "gold_n": len(gold)}
        if len(pred) > len(gold):
            over_seg.append(entry)
        elif len(pred) < len(gold):
            under_seg.append(entry)
        else:
            perfect.append(entry)

    total = len(over_seg) + len(under_seg) + len(perfect)
    return {
        "over_seg_rate":  round(len(over_seg)  / total, 4) if total else 0.0,
        "under_seg_rate": round(len(under_seg) / total, 4) if total else 0.0,
        "perfect_rate":   round(len(perfect)   / total, 4) if total else 0.0,
        "over_count":     len(over_seg),
        "under_count":    len(under_seg),
        "perfect_count":  len(perfect),
        "over_examples":  over_seg[:n_examples],
        "under_examples": under_seg[:n_examples],
    }


# ═════════════════════════════════════════════════════════════════════════════
# Gold-standard average tokens per wordtype (reference baseline)
# ═════════════════════════════════════════════════════════════════════════════

def gold_avg_tokens_per_wordtype(db: SegmentationSource, words: list[str]) -> float:
    """Average number of gold morphemes per word, for words present in db."""
    counts = [len(db.segmentation[w]) for w in words if w in db.segmentation]
    return round(sum(counts) / len(counts), 4) if counts else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Entry points that bundle several metrics together
# ═════════════════════════════════════════════════════════════════════════════

def run_morpholex_metrics(tokenizer: BaseTokenizer, test_words: list[str],
                          morpholex_db: "MorphoLexDB",
                          metrics: set[str] | None = None) -> dict:
    """Run affix_detection / segmentation_consistency (MorphoLexDB-only metrics)."""
    if metrics is None:
        metrics = {"affix_detection", "consistency"}

    results: dict = {"vocab_size": tokenizer.vocab_size()}
    if "affix_detection" in metrics:
        results["affix_detection"] = affix_detection(tokenizer, test_words, morpholex_db)
    if "consistency" in metrics:
        results["consistency"] = segmentation_consistency(tokenizer, test_words, morpholex_db)
    return results


def run_goldstd_metrics(tokenizer: BaseTokenizer, algorithm: str, test_words: list[str],
                        goldstd_db: "GoldStdDB",
                        metrics: set[str] | None = None) -> dict:
    """Run boundary_prf / morpheme_recovery_rate / over_under_segmentation (GoldStdDB-only)."""
    if metrics is None:
        metrics = {"boundary_prf", "morpheme_recovery", "over_under"}

    results: dict = {"vocab_size": tokenizer.vocab_size()}
    if "boundary_prf" in metrics:
        results["boundary_prf"] = boundary_prf(tokenizer, algorithm, test_words, goldstd_db)
    if "morpheme_recovery" in metrics:
        results["morpheme_recovery"] = morpheme_recovery_rate(tokenizer, test_words, goldstd_db)
    if "over_under" in metrics:
        results["over_under"] = over_under_segmentation(tokenizer, test_words, goldstd_db)
    return results
