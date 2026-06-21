"""
tokenizer_interface.py
======================
Shared interface for the BPE / WordPiece project.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import re


# Type aliases
Vocab = set[str]
MergeRule = tuple[str, str]
MergeRules = list[MergeRule]
TokenList = list[str]
Corpus = list[str]


def load_corpus(path: str) -> Corpus:
    """Read a corpus file, one sentence per line."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def preprocess(corpus: Corpus, lowercase: bool = True) -> dict[str, int]:
    """
    Convert a corpus into a word-frequency dictionary.
    Characters inside each word are separated by spaces and a </w> end-of-word
    marker is appended.

    Returns:
        {"l o w </w>": 5, "n e w e s t </w>": 3, ...}
    """
    word_freq: dict[str, int] = defaultdict(int)
    for sentence in corpus:
        if lowercase:
            sentence = sentence.lower()
        sentence = re.sub(r"[^a-z\s]", " ", sentence)
        for word in sentence.split():
            char_word = " ".join(list(word)) + " </w>"
            word_freq[char_word] += 1
    return dict(word_freq)


class BaseTokenizer:
    """Base class for BPE and WordPiece tokenizers."""

    def __init__(self):
        self.vocab: Vocab = set()
        self.is_trained: bool = False

    def train(self, corpus: Corpus, vocab_size: int) -> None:
        """Train the tokenizer on a corpus until the target vocabulary size is reached."""
        raise NotImplementedError

    def tokenize(self, text: str) -> TokenList:
        """Split text into a list of tokens. Unknown subwords are returned as "[UNK]"."""
        raise NotImplementedError

    def tokenize_corpus(self, corpus: Corpus) -> list[TokenList]:
        """Tokenize every sentence in the corpus."""
        return [self.tokenize(sentence) for sentence in corpus]

    def vocab_size(self) -> int:
        return len(self.vocab)

    def contains(self, token: str) -> bool:
        return token in self.vocab


@dataclass
class TokenizationResult:
    """Result of a single tokenization, used for cross-tokenizer comparison."""
    algorithm: str
    original:  str
    tokens:    TokenList
    unk_count: int = field(init=False)

    def __post_init__(self):
        self.unk_count = self.tokens.count("[UNK]")

    def token_count(self) -> int:
        return len(self.tokens)

    def unk_rate(self) -> float:
        if not self.tokens:
            return 0.0
        return self.unk_count / len(self.tokens)

    def __repr__(self):
        return (f"[{self.algorithm}] '{self.original}'\n"
                f"  -> {self.tokens}\n"
                f"  tokens={self.token_count()}, UNK={self.unk_count}")


def compare_tokenizations(
    text: str,
    tokenizers: list[BaseTokenizer],
    names: list[str],
) -> list[TokenizationResult]:
    """Tokenize the same text with multiple tokenizers and return the results."""
    return [
        TokenizationResult(name, text, tok.tokenize(text))
        for tok, name in zip(tokenizers, names)
    ]


def vocab_overlap(vocab_a: Vocab, vocab_b: Vocab) -> dict:
    """
    Compare two vocabularies.

    Returns:
        {
            "only_in_a": {...},
            "only_in_b": {...},
            "shared":    {...},
            "jaccard":   0.42,
        }
    """
    shared = vocab_a & vocab_b
    union  = vocab_a | vocab_b
    return {
        "only_in_a": vocab_a - vocab_b,
        "only_in_b": vocab_b - vocab_a,
        "shared":    shared,
        "jaccard":   len(shared) / len(union) if union else 0.0,
    }


def avg_tokens_per_word(tokenizer: BaseTokenizer, words: list[str]) -> float:
    """Average number of tokens per word (lower means a more complete vocabulary)."""
    total = sum(len(tokenizer.tokenize(w)) for w in words)
    return total / len(words) if words else 0.0
