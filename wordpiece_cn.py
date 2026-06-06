"""
wordpiece_cn.py
===============
WordPiece 分词器 —— 逐行中文注释版本。
代码与 wordpiece.py 完全一致，注释重点标注与 BPE (bpe.py) 的区别。

【与BPE的四个核心区别】
  ① 预处理：词内非首字符加 "##" 前缀，而非词尾加 "</w>"
  ② 打分：score(A,B) = freq(AB) / (freq(A)*freq(B))，BPE 直接用 freq(AB)
  ③ 合并：新 token = A + B[2:] (若B以"##"开头)，BPE 直接拼 A+B
  ④ 推断：贪心最长匹配（greedy longest-match），BPE 按 merge 顺序依次应用
"""

from __future__ import annotations
import heapq
import re
from collections import defaultdict, Counter

# 与 BPE 导入完全相同的基类和类型
from tokenizer_interface import (
    BaseTokenizer,
    Vocab,
    TokenList,
    Corpus,
    MergeRule,
)


# ─────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────

def wp_preprocess(corpus: Corpus, lowercase: bool = True) -> dict[str, int]:
    """
    【与BPE区别①】WordPiece 专用预处理，取代 tokenizer_interface.preprocess()。

    BPE 预处理：  "low" -> "l o w </w>"   （词尾加 </w> 标记词边界）
    WP  预处理：  "low" -> "l ##o ##w"    （词内非首字符加 ## 标记连续性）

    Args:
        corpus: 句子列表
    Returns:
        word_freq: {"l ##o ##w": 3, "n ##e ##w ##e ##s ##t": 2, ...}
    """
    word_freq: dict[str, int] = defaultdict(int)
    for sentence in corpus:
        if lowercase:
            sentence = sentence.lower()
        # 与 BPE 相同：把标点替换成空格，只保留小写字母和空格。
        # 注意：必须替换成空格而非直接删除，否则破折号两侧的词会被拼接
        # 例如 "without—within" 直接删除 → "withoutwithin"（错误的单个词）
        #       "without—within" 替换空格 → "without within"（正确的两个词）
        sentence = re.sub(r"[^a-z\s]", " ", sentence)
        for word in sentence.split():
            chars = list(word)
            if not chars:
                continue
            # 【与BPE区别①】第一个字符不加前缀；其余字符加 "##" 前缀
            # BPE 的做法是在末尾拼 " </w>"
            wp_chars = [chars[0]] + ["##" + c for c in chars[1:]]
            word_freq[" ".join(wp_chars)] += 1
    return dict(word_freq)


def get_token_freq(word_freq: dict[str, int]) -> dict[str, int]:
    """
    统计每个 token 在语料中出现的总次数（按词频加权）。

    WordPiece 打分公式的分母需要单个 token 的频率，BPE 不需要。
    BPE 只需要 pair 的频率（get_stats）。

    Returns:
        {"l": 4, "##o": 4, "##w": 5, ...}
    """
    token_freq: dict[str, int] = defaultdict(int)
    for word, freq in word_freq.items():
        for tok in word.split():
            token_freq[tok] += freq  # 每个 token 计入其词的频率
    return dict(token_freq)


def get_pair_freq(word_freq: dict[str, int]) -> dict[tuple[str, str], int]:
    """
    统计相邻 token 对的频率（与 BPE 的 get_stats 逻辑完全相同）。

    Returns:
        {("l","##o"): 4, ("##o","##w"): 4, ...}
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
    【与BPE区别②】计算 WordPiece 打分。

    WordPiece:   score(A, B) = freq(AB) / (freq(A) * freq(B))
    BPE:         直接用 freq(AB)，无归一化

    直觉：WP 偏好"几乎总是一起出现"的对，而不是仅因为组成词很常见而频率高的对。
    例如，"##s" 和 "##t" 在语料中几乎每次 "##s" 出现都紧跟 "##t"，
    所以 score(##s, ##t) 高；而 ("l","##o") 虽然出现次数多，
    但 "l" 也单独出现（词首），所以得分较低。

    Returns:
        scores: {(A, B): float, ...}
    """
    scores: dict[tuple[str, str], float] = {}
    for (a, b), ab_freq in pair_freq.items():
        fa = token_freq.get(a, 0)
        fb = token_freq.get(b, 0)
        if fa > 0 and fb > 0:
            # 【与BPE区别②】除以两个 token 各自频率的乘积
            scores[(a, b)] = ab_freq / (fa * fb)
    return scores


def wp_merge_vocab(
    pair: tuple[str, str],
    word_freq: dict[str, int],
) -> dict[str, int]:
    """
    【与BPE区别③】合并时去掉 B 的 "##" 前缀，BPE 直接拼接 A+B。

    合并规则：
        new_token = A + B[2:]  （当 B 以 "##" 开头时去掉前缀）
        new_token = A + B      （其他情况，理论上不会出现）

    示例（对比 BPE 直接拼接）：
        BPE:  ("l", "o")    -> "lo"    (直接拼)
        WP:   ("l", "##o")  -> "lo"    (A不含##，B去掉##后拼)

        BPE:  ("##e", "##s")  -> "##e##s"  ← BPE 里根本没有 ## 前缀
        WP:   ("##e", "##s")  -> "##es"    (A保留##，B去掉##)
    """
    a, b = pair
    # 【与BPE区别③】新 token 的构造方式
    new_tok = a + (b[2:] if b.startswith("##") else b)
    new_word_freq: dict[str, int] = {}
    for word, freq in word_freq.items():
        tokens = word.split()
        merged: list[str] = []
        i = 0
        n = len(tokens)
        while i < n:
            # 与 BPE 相同的左到右扫描合并逻辑，非重叠
            if i < n - 1 and tokens[i] == a and tokens[i + 1] == b:
                merged.append(new_tok)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        new_word_freq[" ".join(merged)] = freq
    return new_word_freq


# ─────────────────────────────────────────
# WordPiece 分词器
# ─────────────────────────────────────────

class WordPieceTokenizer(BaseTokenizer):
    """
    WordPiece 分词器，继承 BaseTokenizer（与 BPETokenizer 相同的基类）。

    self.vocab:  set[str]          训练得到的词汇表
    self.merges: list[(str,str)]   有序合并规则列表（与 BPE 结构相同）

    公共接口与 BPETokenizer 完全一致：
        train(corpus, vocab_size, *, fast, verbose)
        tokenize(text) -> TokenList
        tokenize_corpus / vocab_size / contains  （继承自 BaseTokenizer）
    """

    def __init__(self):
        super().__init__()  # 初始化 self.vocab = set(), self.is_trained = False
        self.merges: list[MergeRule] = []  # 与 BPE 完全相同的结构

    # ─────────────────────────────────────────
    # 训练入口（接口与 BPE 完全一致）
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
        训练 WordPiece，直到词汇表大小达到 vocab_size。

        fast=True  -> 堆 + 增量更新（与 BPE fast 思路相同，但额外维护 token_freq）
        fast=False -> Naive：每轮从头重新计算所有分数

        两种模式产生完全相同的 self.vocab 和 self.merges。
        """
        if fast:
            self._train_fast(corpus, vocab_size, verbose=verbose)
        else:
            self._train_naive(corpus, vocab_size, verbose=verbose)

    # ── Naive 训练 ──────────────────────────────────────────────────────
    def _train_naive(self, corpus: Corpus, vocab_size: int, verbose: bool = False) -> None:
        # 【与BPE区别①】使用 wp_preprocess 而非 preprocess
        word_freq = wp_preprocess(corpus)

        # 初始词汇表 = 所有单字符 token（含 ## 前缀版本）
        self.vocab = set()
        for word in word_freq:
            for tok in word.split():
                self.vocab.add(tok)
        self.merges = []

        if verbose:
            print(f"[WP naive init] vocab_size={len(self.vocab)}")

        while len(self.vocab) < vocab_size:
            # 【与BPE区别②】每轮都需要统计单 token 频率（BPE 不需要）
            token_freq = get_token_freq(word_freq)
            pair_freq  = get_pair_freq(word_freq)
            if not pair_freq:
                break
            # 【与BPE区别②】用 WP 打分，而非直接用 pair_freq
            scores = get_wp_scores(token_freq, pair_freq)
            if not scores:
                break

            # 选最高分；并列时取字典序最小的 pair（与 BPE 并列规则一致）
            best_pair  = min(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            best_score = scores[best_pair]

            # 合并并更新 word_freq
            word_freq  = wp_merge_vocab(best_pair, word_freq)
            a, b       = best_pair
            # 【与BPE区别③】新 token 去掉 B 的 "##" 前缀
            new_token  = a + (b[2:] if b.startswith("##") else b)
            self.vocab.add(new_token)
            self.merges.append(best_pair)

            if verbose:
                # 注意打印的是 score（浮点数），BPE 打印的是 freq（整数）
                print(f"[WP naive merge {len(self.merges):>3}] "
                      f"{best_pair} (score={best_score:.6f}) -> {new_token!r}")

        self.is_trained = True

    # ── Fast 训练 ────────────────────────────────────────────────────────
    def _train_fast(self, corpus: Corpus, vocab_size: int, verbose: bool = False) -> None:
        """
        与 BPE fast 结构相同，额外增加对 token_freq 的增量维护。

        【与BPE fast 的区别】
        - 额外维护 token_freq[t]：每个 token 的当前全局频率
          （BPE fast 不需要，因为 BPE 打分只用 pair_freq）
        - 额外维护 token_pairs[t]：涉及 token t 的所有 pair 的集合
          当 token_freq[t] 变化时，用此索引找出所有需要刷新分数的 pair
        - 堆条目格式：(-score, pair, pf_snap, tfa_snap, tfb_snap)
          比 BPE 的 (-freq, pair) 多了三个快照值
        - 弹堆验证：三个快照值全部匹配当前值才认为未过期
          （BPE 只需验证 pair_freq 一个值）

        与 BPE fast 相同的部分：
        - words[i] / freqs[i] 并行数组
        - pair_where[p]：倒排索引（包含 pair p 的词的下标集合）
        - 懒惰堆（stale 条目在弹出时跳过，不主动删除）
        - _apply_one_merge 作为唯一合并语义来源
        """
        # 【与BPE区别①】用 wp_preprocess
        word_freq = wp_preprocess(corpus)

        # 拆成并行数组 words / freqs，初始化词汇表
        words: list[list[str]] = []
        freqs: list[int]       = []
        self.vocab = set()
        for w, f in word_freq.items():
            toks = w.split()
            words.append(toks)
            freqs.append(f)
            self.vocab.update(toks)
        self.merges = []

        # ── 初始化增量数据结构 ──
        pair_freq:   dict[tuple[str, str], int]            = defaultdict(int)
        pair_where:  dict[tuple[str, str], set[int]]       = defaultdict(set)
        token_freq:  dict[str, int]                        = defaultdict(int)   # 【WP新增】
        token_pairs: dict[str, set[tuple[str, str]]]       = defaultdict(set)   # 【WP新增】

        for wi, toks in enumerate(words):
            f = freqs[wi]
            for tok in toks:
                token_freq[tok] += f                  # 【WP新增】统计单 token 频率
            for a, b in zip(toks, toks[1:]):
                p = (a, b)
                pair_freq[p]  += f
                pair_where[p].add(wi)

        # 构建 token_pairs 索引（token -> 涉及它的所有 pair）
        for p in list(pair_freq):
            a, b = p
            token_pairs[a].add(p)
            token_pairs[b].add(p)

        # ── 堆辅助函数 ──
        def push_score(p: tuple[str, str]) -> None:
            """将 pair p 的当前分数压入堆，同时记录三个快照值。"""
            a, b  = p
            pf    = pair_freq.get(p, 0)
            tfa   = token_freq.get(a, 0)
            tfb   = token_freq.get(b, 0)
            if pf > 0 and tfa > 0 and tfb > 0:
                # 【与BPE区别②】堆存 -score（WP打分），BPE 存 -freq
                # 额外存三个快照用于弹堆时验证是否过期
                heapq.heappush(heap, (-pf / (tfa * tfb), p, pf, tfa, tfb))

        # 构建初始堆
        heap: list = []
        for p in pair_freq:
            push_score(p)
        heapq.heapify(heap)

        if verbose:
            print(f"[WP fast init] vocab_size={len(self.vocab)}, pairs={len(pair_freq)}")

        # ── 主循环 ──
        while len(self.vocab) < vocab_size:

            # (1) 弹出最优 pair，跳过过期条目
            best_pair = None
            while heap:
                neg_score, p, pf_snap, tfa_snap, tfb_snap = heapq.heappop(heap)
                a, b = p
                # 【与BPE区别】WP 需要验证三个快照；BPE 只验证 pair_freq 一个
                if (pair_freq.get(p, 0)    == pf_snap  and
                        token_freq.get(a, 0) == tfa_snap and
                        token_freq.get(b, 0) == tfb_snap and
                        pf_snap > 0):
                    best_pair = p
                    break
            if best_pair is None:
                break

            a, b       = best_pair
            best_score = pair_freq[best_pair] / (token_freq[a] * token_freq[b])
            # 【与BPE区别③】新 token 去掉 B 的 "##"
            new_tok    = a + (b[2:] if b.startswith("##") else b)
            self.vocab.add(new_tok)
            self.merges.append(best_pair)

            # (2) 从活跃结构中移除已合并的 pair
            affected = pair_where.pop(best_pair, set())
            pair_freq.pop(best_pair, None)
            token_pairs[a].discard(best_pair)
            token_pairs[b].discard(best_pair)

            # (3) 处理受影响的词，收集需要刷新分数的 pair
            score_refresh: set[tuple[str, str]] = set()

            for wi in affected:
                f        = freqs[wi]
                old_toks = words[wi]

                old_pair_counts = Counter(zip(old_toks, old_toks[1:]))
                old_tok_counts  = Counter(old_toks)

                # 使用 _apply_one_merge 保证与推断阶段语义一致（与 BPE 相同）
                new_toks = self._apply_one_merge(old_toks, best_pair)
                words[wi] = new_toks

                new_pair_counts = Counter(zip(new_toks, new_toks[1:]))
                new_tok_counts  = Counter(new_toks)

                # 差分更新 pair_freq 和 pair_where（与 BPE fast 逻辑相同）
                for p in set(old_pair_counts) | set(new_pair_counts):
                    pa, pb = p
                    delta_p = (new_pair_counts.get(p, 0) - old_pair_counts.get(p, 0)) * f
                    if delta_p != 0:
                        pair_freq[p] = pair_freq.get(p, 0) + delta_p
                        score_refresh.add(p)       # pair_freq 变了，分子变了
                    if new_pair_counts.get(p, 0) > 0:
                        pair_where[p].add(wi)
                        token_pairs[pa].add(p)     # 维护 token_pairs 索引
                        token_pairs[pb].add(p)
                    else:
                        pair_where[p].discard(wi)

                # 【WP新增】差分更新 token_freq
                for t in set(old_tok_counts) | set(new_tok_counts):
                    delta_t = (new_tok_counts.get(t, 0) - old_tok_counts.get(t, 0)) * f
                    if delta_t != 0:
                        token_freq[t] = token_freq.get(t, 0) + delta_t
                        if token_freq.get(t, 0) <= 0:
                            token_freq.pop(t, None)
                        # token_freq[t] 变了 -> 所有含 t 的 pair 分母都变了
                        # 这是 WP fast 比 BPE fast 多出的刷新来源
                        score_refresh |= set(token_pairs.get(t, set()))

            # (4) 为所有需要刷新的 pair 推入新分数（懒惰更新）
            for p in score_refresh:
                if pair_freq.get(p, 0) > 0:
                    push_score(p)

            if verbose:
                print(f"[WP fast merge {len(self.merges):>3}] "
                      f"{best_pair} (score={best_score:.6f}) -> {new_tok!r}")

        self.is_trained = True

    # ─────────────────────────────────────────
    # 推断：tokenize
    # ─────────────────────────────────────────
    def tokenize(self, text: str) -> TokenList:
        """
        【与BPE区别④】标准 WordPiece 贪心最长匹配推断。

        对每个词从左到右逐步找词汇表中能匹配的最长前缀：
        - 非词首片段在查找时加 "##" 前缀
        - 若某个位置连单字符都找不到匹配 → 整词返回 ["[UNK]"]（BERT 标准行为）

        【旧版问题及修复】
        旧版按 self.merges 顺序依次应用（与 BPE 推断相同），存在两个缺陷：
          缺陷1：merge 顺序推断可能错过词汇表中已有的更长子词
          缺陷2：OOV 以 token 为粒度替换，导致 "unknown" →
                 ['[UNK]', '[UNK]', '[UNK]', '[UNK]', '##o', '##w', '[UNK]']
                 已知字符夹杂在 [UNK] 中，不符合标准行为
          缺陷3：旧版通过 wp_preprocess 迭代，该函数对词去重，
                 导致多词文本中重复的词只被 tokenize 一次

        新版修复：
          - 用 greedy longest-match 直接在词汇表搜索（BERT/HuggingFace 标准）
          - 整词级别 OOV：任意位置匹配失败 → 整词 ["[UNK]"]
          - 直接 text.split() 保留原始词序，修复去重 bug

        【与BPE区别】输出 token 含 "##" 前缀（BPE 输出含 "</w>" 后缀）
            BPE:   "newest" -> ["newest</w>"]
            WP:    "newest" -> ["newest"]（整词在词汇表中时）
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer is not trained yet; call train() first.")

        # 预处理与训练时保持一致：小写 + 非字母替换为空格
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        all_tokens: TokenList = []
        # 直接按原始顺序处理每个词，不去重（修复旧版 wp_preprocess 去重问题）
        for word in text.split():
            all_tokens.extend(self._tokenize_word(word))
        return all_tokens

    def _tokenize_word(self, word: str) -> TokenList:
        """
        对单个词做贪心最长匹配分词，返回子词列表。

        算法步骤：
            start = 0
            while start < len(word):
                end = len(word)        # 从最长前缀开始尝试
                while start < end:
                    substr = word[start:end]
                    if start > 0: substr = "##" + substr   # 非词首加前缀
                    if substr in self.vocab: break         # 命中，记录并推进
                    end -= 1                               # 缩短，继续尝试
                if end == start:                           # 连单字符都没命中
                    return ["[UNK]"]                       # 整词失败
                tokens.append(substr); start = end

        为什么不能用旧版（按 merge 顺序应用）？
            merge 顺序反映训练时的合并历史，推断时某些更长子词可能已在词汇表
            但因为所需的中间合并步骤顺序不同而被跳过，greedy search 可直接找到。

        OOV 处理（整词 vs 逐 token）：
            标准 BERT WordPiece：只要有一个位置无法匹配（哪怕单字符），
            整个词返回 ["[UNK]"]，而不是把失败位置替换成 "[UNK]" 继续处理。
        """
        tokens: TokenList = []
        start = 0
        n = len(word)
        while start < n:
            end = n
            cur_substr = None
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr   # 非词首片段加 ## 前缀后查词汇表
                if substr in self.vocab:
                    cur_substr = substr      # 命中：记录当前子词
                    break
                end -= 1                     # 未命中：缩短前缀，重试
            if cur_substr is None:
                # 该位置连单字符都不在词汇表 → 整词是 OOV
                return ["[UNK]"]
            tokens.append(cur_substr)
            start = end                      # 从匹配结束处继续
        return tokens

    @staticmethod
    def _apply_one_merge(tokens: TokenList, pair: tuple[str, str]) -> TokenList:
        """
        对 token 列表应用单条合并规则，左到右扫描、非重叠。

        【与BPE区别③】新 token 构造：去掉 B 的 "##" 前缀
        BPE: new_tok = a + b        （直接拼接）
        WP:  new_tok = a + b[2:]    （B 以 "##" 开头时去掉前缀）

        控制流（while 循环、非重叠语义）与 BPE 完全相同。
        训练（fast 版）和推断共用此方法，保证语义一致。
        """
        a, b    = pair
        # 【与BPE区别③】
        new_tok = a + (b[2:] if b.startswith("##") else b)
        i       = 0
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
        """按顺序应用 self.merges 中的全部规则（与 BPE 完全相同）。"""
        for pair in self.merges:
            tokens = self._apply_one_merge(tokens, pair)
        return tokens


# ─────────────────────────────────────────
# 自测（与 wordpiece.py 相同）
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("Block 1: wp_preprocess + WP 打分")
    print("=" * 60)

    corpus = ["low low low lowest newest"]
    wf = wp_preprocess(corpus)
    print("wp_preprocess 输出（对比 BPE 的 'l o w </w>' 格式）：")
    for k, v in sorted(wf.items()):
        print(f"  {k!r}: {v}")

    tf = get_token_freq(wf)
    pf = get_pair_freq(wf)
    sc = get_wp_scores(tf, pf)

    print("\nToken 频率（BPE 无此统计）：")
    for tok, cnt in sorted(tf.items()):
        print(f"  {tok!r}: {cnt}")

    print("\nPair 得分（WP: freq(AB)/(freq(A)*freq(B))，BPE: freq(AB)）：")
    for pair, score in sorted(sc.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {pair}: score={score:.6f}  "
              f"[pair_freq={pf[pair]}, freq_A={tf[pair[0]]}, freq_B={tf[pair[1]]}]")

    a, b = "##s", "##t"
    expected_score = pf[(a, b)] / (tf[a] * tf[b])
    assert abs(sc[(a, b)] - expected_score) < 1e-12
    print(f"\n[OK] score({(a,b)}) = {pf[(a,b)]}/({tf[a]}*{tf[b]}) = {expected_score:.6f}")

    print("\n" + "=" * 60)
    print("Block 2: Naive WordPiece 训练")
    print("=" * 60)

    wp = WordPieceTokenizer()
    wp.train(corpus, vocab_size=15, fast=False, verbose=True)
    print(f"\n最终词汇表 ({len(wp.vocab)}): {sorted(wp.vocab)}")
    print(f"合并规则 ({len(wp.merges)}): {wp.merges}")

    print("\n" + "=" * 60)
    print("Block 3: tokenize")
    print("=" * 60)
    for w in ["lowest", "newest", "low", "newer", "unknown"]:
        print(f"  tokenize({w!r:>10}) -> {wp.tokenize(w)}")

    print("\n" + "=" * 60)
    print("Block 4: Fast == Naive 等价性 + 速度对比")
    print("=" * 60)

    naive = WordPieceTokenizer(); naive.train(corpus, vocab_size=15, fast=False)
    fast  = WordPieceTokenizer(); fast.train(corpus,  vocab_size=15, fast=True)
    assert naive.vocab  == fast.vocab
    assert naive.merges == fast.merges
    print("[OK] naive.vocab == fast.vocab 且 naive.merges == fast.merges")

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

    t0 = time.perf_counter()
    n2 = WordPieceTokenizer(); n2.train(medium_corpus, vocab_size=target_vocab, fast=False)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    f2 = WordPieceTokenizer(); f2.train(medium_corpus, vocab_size=target_vocab, fast=True)
    t_fast = time.perf_counter() - t0

    assert n2.vocab  == f2.vocab
    assert n2.merges == f2.merges
    speedup = t_naive / t_fast if t_fast > 0 else float("inf")
    print(f"\n  中等语料: {len(medium_corpus)} 句 x 20 词, target_vocab={target_vocab}")
    print(f"  Naive: {t_naive*1000:7.1f} ms")
    print(f"  Fast : {t_fast*1000:7.1f} ms")
    print(f"  [OK] vocab/merges 完全一致；Fast 加速约 {speedup:.1f}x")

    print("\n" + "=" * 60)
    print("Block 5: BPE vs WordPiece 对比（同一语料）")
    print("=" * 60)
    try:
        from bpe import BPETokenizer
        bpe_tok = BPETokenizer()
        bpe_tok.train(corpus, vocab_size=15, fast=False)
        wp_tok  = WordPieceTokenizer()
        wp_tok.train(corpus,  vocab_size=15, fast=False)
        print(f"{'词':>12}  {'BPE':30}  {'WordPiece':30}")
        print("-" * 78)
        for w in ["lowest", "newest", "low", "newer", "unknown"]:
            b = str(bpe_tok.tokenize(w))
            p = str(wp_tok.tokenize(w))
            print(f"  {w!r:>10}  {b:30}  {p:30}")
        print("\nBPE 合并顺序:       ", bpe_tok.merges)
        print("WordPiece 合并顺序: ", wp_tok.merges)
        print("\n注意区别：")
        print("  BPE 优先合并高频对（low/lowest 里的 l+o 先合并）")
        print("  WP  优先合并高分对（##s+##t 因为几乎总是同时出现而先合并）")
    except ImportError:
        print("(bpe.py 未找到，跳过对比)")

    # ── Block 6: 真实语料测试 ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Block 6: WordPiece 在 data/test_bpe.txt 上训练")
    print("=" * 60)

    import os
    # 用脚本所在目录拼接路径，避免工作目录不同导致找不到文件
    corpus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_bpe.txt")
    with open(corpus_path, encoding="utf-8") as _f:
        # 去掉空行，每行作为一句话送入语料库
        real_corpus = [line.strip() for line in _f if line.strip()]

    target_vocab = 500
    print(f"语料: {len(real_corpus)} 行  |  目标词汇表大小: {target_vocab}")

    # 使用 fast=True（堆 + 增量更新）加速训练
    wp_real = WordPieceTokenizer()
    wp_real.train(real_corpus, vocab_size=target_vocab, fast=True)
    print(f"词汇表大小: {len(wp_real.vocab)}  |  学到的合并规则数: {len(wp_real.merges)}")

    # 最长 token 反映了算法学到了哪些有意义的长子词
    long_toks = sorted(wp_real.vocab, key=len, reverse=True)[:15]
    print(f"最长的 token（前 15 个）: {long_toks}")

    print("\n示例分词结果（使用贪心最长匹配）：")
    test_words = ["harpooneer", "landlord", "sleeping", "whale", "cannibal", "unknown", "whaling"]
    for w in test_words:
        print(f"  {w!r:>14} -> {wp_real.tokenize(w)}")
