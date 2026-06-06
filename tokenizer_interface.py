"""
tokenizer_interface.py
======================
公共接口定义 - BPE / WordPiece 项目
三人共用，A和B各自实现，C直接调用。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator
from collections import defaultdict
import re


# ─────────────────────────────────────────
# 1. 数据结构约定
# ─────────────────────────────────────────

# 词汇表：就是一个字符串集合
Vocab = set[str]

# 合并规则列表（BPE专用，按顺序应用）
MergeRule = tuple[str, str]          # e.g. ("p", "p")
MergeRules = list[MergeRule]

# tokenize的输出：token字符串列表
TokenList = list[str]

# 语料库：句子列表
Corpus = list[str]


# ─────────────────────────────────────────
# 2. 预处理（C写，A和B都调用这个）
# ─────────────────────────────────────────

def load_corpus(path: str) -> Corpus:
    """从文件读取语料库，每行一句话，返回句子列表。"""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def preprocess(corpus: Corpus, lowercase: bool = True) -> dict[str, int]:
    """
    把语料库转成词频字典。
    每个词内部用空格隔开字符，BPE还会加</w>词尾标记。

    返回：
        word_freq: {"l o w </w>": 5, "n e w e s t </w>": 3, ...}
    """
    word_freq: dict[str, int] = defaultdict(int)
    for sentence in corpus:
        if lowercase:
            sentence = sentence.lower()
        # 去掉标点，只保留字母和空格
        sentence = re.sub(r"[^a-z\s]", "", sentence)
        for word in sentence.split():
            # 字符之间加空格，词尾加</w>
            char_word = " ".join(list(word)) + " </w>"
            word_freq[char_word] += 1
    return dict(word_freq)


# ─────────────────────────────────────────
# 3. 抽象基类（A和B各自继承实现）
# ─────────────────────────────────────────

class BaseTokenizer(ABC):
    """
    BPE和WordPiece都继承这个类。
    C只需要用这个类里定义的方法，不关心内部实现。
    """

    def __init__(self):
        self.vocab: Vocab = set()
        self.is_trained: bool = False

    @abstractmethod
    def train(self, corpus: Corpus, vocab_size: int) -> None:
        """
        在语料库上训练，建立词汇表。

        参数：
            corpus:     句子列表
            vocab_size: 目标词汇表大小（比如 1000, 8000, 32000）
        """
        ...

    @abstractmethod
    def tokenize(self, text: str) -> TokenList:
        """
        把一个字符串切分成token列表。
        未知子词用 "[UNK]" 表示。

        参数：
            text: 一句话或一个词
        返回：
            ["un", "##happy", "##ness"]  # WordPiece风格
            ["un", "happiness"]           # BPE风格
        """
        ...

    def tokenize_corpus(self, corpus: Corpus) -> list[TokenList]:
        """对整个语料库的每句话都tokenize，返回列表的列表。"""
        return [self.tokenize(sentence) for sentence in corpus]

    def vocab_size(self) -> int:
        """返回当前词汇表大小。"""
        return len(self.vocab)

    def contains(self, token: str) -> bool:
        """判断某个token是否在词汇表里。"""
        return token in self.vocab


# ─────────────────────────────────────────
# 4. 评估用的辅助函数（C来实现具体逻辑）
# ─────────────────────────────────────────

@dataclass
class TokenizationResult:
    """存一次tokenize的结果，方便C做对比。"""
    algorithm: str               # "BPE" 或 "WordPiece"
    original:  str               # 原始文本
    tokens:    TokenList         # tokenize结果
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
                f"  → {self.tokens}\n"
                f"  tokens={self.token_count()}, UNK={self.unk_count}")


def compare_tokenizations(
    text: str,
    tokenizers: list[BaseTokenizer],
    names: list[str]
) -> list[TokenizationResult]:
    """
    用多个tokenizer对同一段文本tokenize，返回结果列表，方便C对比。

    用法（C的代码里）：
        results = compare_tokenizations(
            "unhappiness",
            [bpe, wordpiece],
            ["BPE", "WordPiece"]
        )
        for r in results:
            print(r)
    """
    return [
        TokenizationResult(name, text, tok.tokenize(text))
        for tok, name in zip(tokenizers, names)
    ]


def vocab_overlap(vocab_a: Vocab, vocab_b: Vocab) -> dict:
    """
    计算两个词汇表的重叠情况。

    返回：
        {
            "only_in_a": {...},
            "only_in_b": {...},
            "shared":    {...},
            "jaccard":   0.42      # 交集/并集
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
    """
    计算平均每个词被切成几个token（越少说明词汇表越"完整"）。
    """
    total = sum(len(tokenizer.tokenize(w)) for w in words)
    return total / len(words) if words else 0.0


# ─────────────────────────────────────────
# 5. 使用示例（三人都可以参考）
# ─────────────────────────────────────────

if __name__ == "__main__":

    # ── A的代码（bpe.py里）会长这样 ──
    # from tokenizer_interface import BaseTokenizer, preprocess
    # class BPETokenizer(BaseTokenizer):
    #     def train(self, corpus, vocab_size): ...
    #     def tokenize(self, text): ...

    # ── B的代码（wordpiece.py里）会长这样 ──
    # from tokenizer_interface import BaseTokenizer, preprocess
    # class WordPieceTokenizer(BaseTokenizer):
    #     def train(self, corpus, vocab_size): ...
    #     def tokenize(self, text): ...

    # ── C的代码（evaluate.py里）会长这样 ──
    # from bpe import BPETokenizer
    # from wordpiece import WordPieceTokenizer
    # from tokenizer_interface import compare_tokenizations, vocab_overlap
    #
    # corpus = load_corpus("data/corpus.txt")
    #
    # bpe = BPETokenizer()
    # bpe.train(corpus, vocab_size=1000)
    #
    # wp = WordPieceTokenizer()
    # wp.train(corpus, vocab_size=1000)
    #
    # results = compare_tokenizations(
    #     "unhappiness",
    #     [bpe, wp],
    #     ["BPE", "WordPiece"]
    # )
    # for r in results:
    #     print(r)
    #
    # overlap = vocab_overlap(bpe.vocab, wp.vocab)
    # print(f"Jaccard similarity: {overlap['jaccard']:.2%}")

    print("接口定义文件，直接import使用。")
