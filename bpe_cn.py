"""
bpe_cn.py  ——  内部学习版（中文事无巨细注释）
================================================
本文件和 bpe.py 的代码逻辑**完全一样**，区别只是注释更详尽，
方便我和队友逐行理解发生了什么。对外（交给老师 / 给 C 调用）请用 bpe.py。

文件整体结构（自顶向下）：
    1. 顶层辅助函数        get_stats / merge_vocab
    2. 类 BPETokenizer
        2a. __init__          初始化空 vocab 和空 merges
        2b. train             公开入口；按 fast 开关分派
        2c. _train_naive      朴素实现（O(n²) 量级）
        2d. _train_fast       带 heapq 和增量更新的快速实现
        2e. tokenize          推理：把一段文本切成 token
        2f. _apply_one_merge  单条 merge 规则的合并函数（核心原子操作）
        2g. _apply_merges     依次跑 self.merges 里所有规则
    3. if __name__ == "__main__":  自测 + benchmark

关键约定（来自 tokenizer_interface.py，不许动）：
    - Vocab     = set[str]
    - TokenList = list[str]
    - Corpus    = list[str]  （每个元素是一句话）
    - MergeRule = tuple[str, str]
    - preprocess(corpus) 会做：小写化 + 去标点 + 每个词字符间加空格 +
      词尾加 "</w>"，返回 dict[str,int]，形如 {"l o w </w>": 3, ...}
    - 抽象基类 BaseTokenizer 要求子类实现 train() 和 tokenize()
    - 未知子词输出 "[UNK]"
"""

from __future__ import annotations
import heapq                                # 标准库的堆，用来在 Fast 版本里 O(log n) 取最大频次 pair
from collections import defaultdict, Counter  # defaultdict 用于 pair_freq / pair_where；Counter 用于差分统计

from tokenizer_interface import (
    BaseTokenizer,   # 抽象基类，必须继承
    preprocess,      # 预处理函数（已写好）
    Vocab,           # = set[str]，纯类型别名
    TokenList,       # = list[str]
    Corpus,          # = list[str]
    MergeRule,       # = tuple[str, str]
)


# ─────────────────────────────────────────────────────────────────
# 1. 顶层辅助函数
# ─────────────────────────────────────────────────────────────────

def get_stats(word_freq: dict[str, int]) -> dict[tuple[str, str], int]:
    """
    【作用】
        给定 word_freq（每个词 + 该词出现次数），统计所有"相邻 token 对"
        在整个语料里出现了多少次（按词频加权）。

    【输入举例】
        word_freq = {"l o w </w>": 3, "n e w </w>": 2}
        意思是：词 "low" 出现了 3 次，词 "new" 出现了 2 次。
        注意 key 里的 token 已经被空格分开了，词尾的 </w> 也已经加好。

    【输出举例】
        {("l","o"): 3, ("o","w"): 3, ("w","</w>"): 3,
         ("n","e"): 2, ("e","w"): 2, ("w","</w>"): 2}
        说明：(l,o) 这个相邻对来自 "low" 词，所以贡献 +3；
        (w,</w>) 在两个词里都出现，所以是 3+2=5？—— 不对，
        我们的输出里 (w,</w>): 3 是因为我们对每个 key 单独遍历后**累加**到
        同一个 dict 里。也就是：
            "l o w </w>" (×3) 贡献：(w,</w>) += 3
            "n e w </w>" (×2) 贡献：(w,</w>) += 2
        最终 (w,</w>) = 5。我例子里写错了，正确的是 5。
        （这条 docstring 的真实含义是：所有相邻 pair 在整个语料的加权出现次数。）

    【为什么需要这个】
        BPE 训练的每一轮都要找"现在出现频率最高的相邻对"，所以要先算出
        每个 pair 的频率。Naive 版本每轮都重算一次；Fast 版本只在初始化
        算一次，之后做增量更新。
    """
    # defaultdict(int) 的好处：访问不存在的 key 会自动返回 0，可以直接 +=
    pair_freq: dict[tuple[str, str], int] = defaultdict(int)
    for word, freq in word_freq.items():
        # word 形如 "l o w </w>"，用 split() 拆成 ["l","o","w","</w>"]
        tokens = word.split()
        # 遍历相邻对：(tokens[0],tokens[1]), (tokens[1],tokens[2]), ...
        # 注意是 len(tokens) - 1，因为最后一个 token 没有"下一个"
        for i in range(len(tokens) - 1):
            # 这一对在这个词里出现 1 次，所以累加 freq（这个词的词频）
            pair_freq[(tokens[i], tokens[i + 1])] += freq
    # 返回前转成普通 dict，避免外部不小心利用 defaultdict 的副作用
    return dict(pair_freq)


def merge_vocab(
    pair: tuple[str, str],
    word_freq: dict[str, int],
) -> dict[str, int]:
    """
    【作用】
        把指定的 pair=(A, B) 在 word_freq 里的所有"相邻 A B"合并成 "AB"。
        这是 Naive 训练每轮要做的"应用一条 merge 规则"。

    【举例】
        pair = ("l","o")
        输入 word_freq = {"l o w </w>": 3, "n e w </w>": 2}
        输出 word_freq = {"lo w </w>": 3, "n e w </w>": 2}
        （第一个词的 "l o" 被合并成 "lo"，第二个词没有 l-o 相邻所以不变）

    【⚠️ 易错点：千万不能用 str.replace 直接替换】
        看起来很诱人："把 'A B' 替换成 'AB' 不就行了？"
        但这有跨 token 边界的 bug。例子：
            假设训练到某一步，已经有 token "j</w>" 存在，
            某个词的字符串表示是 "l j</w>"（即 token 序列 ['l','j</w>']）。
            现在要应用 merge ("l","j")，理论上应该不变（因为这词里没有
            ['l','j'] 这两个相邻 token —— 'l' 后面跟的是 'j</w>'，不是 'j'）。

            但 "l j</w>".replace("l j", "lj") 会怎样？
            "l j" 这个字符串子串确实在 "l j</w>" 里出现了（前三个字符），
            会被错误替换成 "lj</w>" —— token 边界被破坏！

        所以**正确做法**：先用 split() 把词拆回 token list，然后只在
        相邻 token **完全等于** (A, B) 时合并，最后用空格重新 join。

    【实现】
        和下方 _apply_one_merge 的逻辑完全一样，只是这里的输入/输出是
        字符串形式（word_freq 的 key 是字符串），而 _apply_one_merge
        操作的是 list[str]。两份代码功能等价。
    """
    a, b = pair  # 解包成单独变量，循环里好写
    new_word_freq: dict[str, int] = {}
    for word, freq in word_freq.items():
        tokens = word.split()        # 词的当前 token 列表
        merged: list[str] = []       # 合并后的新 token 列表
        i = 0
        n = len(tokens)
        while i < n:
            # 条件：还有下一个 token & 当前 token == A & 下一个 token == B
            if i < n - 1 and tokens[i] == a and tokens[i + 1] == b:
                merged.append(a + b)  # 把 (A, B) 合并成字符串 "AB"
                i += 2                # 跳过 2 个位置（避免重叠合并）
            else:
                merged.append(tokens[i])
                i += 1
        new_word_freq[" ".join(merged)] = freq  # 再 join 回字符串作为新 key
    return new_word_freq


# ─────────────────────────────────────────────────────────────────
# 2. BPETokenizer 类
# ─────────────────────────────────────────────────────────────────

class BPETokenizer(BaseTokenizer):
    """
    BPE 分词器。继承自 BaseTokenizer，需要实现 train() 和 tokenize()。

    实例属性：
        self.vocab    : set[str]            训练得到的词表（包含初始单字符和合并出来的子词）
        self.merges   : list[(str,str)]     合并规则，**顺序敏感**
                                            —— tokenize 时必须按这个顺序一条条应用
        self.is_trained: bool               基类提供；训练完置 True，给 tokenize 检查用
    """

    def __init__(self):
        # 调基类的 __init__，它会把 self.vocab 设成空 set，self.is_trained 设成 False
        super().__init__()
        # BPE 特有的状态：merges 是一个有序列表，初始为空
        self.merges: list[MergeRule] = []

    # ─────────────────────────────────────────
    # 2b. 训练入口（统一接口）
    # ─────────────────────────────────────────
    def train(
        self,
        corpus: Corpus,
        vocab_size: int,
        *,                       # * 后面的参数必须用关键字传递，避免位置参数歧义
        fast: bool = True,       # 默认走 Fast 路径
        verbose: bool = False,   # 打印每一步 merge 的过程
    ) -> None:
        """
        外部使用者（包括队友 C）只需要调这个方法。

        参数：
            corpus     : list[str]，每个元素是一句话
            vocab_size : 目标词表大小（包括初始单字符）
            fast       : True 用 Fast 实现；False 用 Naive 实现
            verbose    : 是否打印每一步合并

        保证：fast=True 和 fast=False 的产出（vocab、merges 的内容和顺序）
              **完全一致**（同频破平规则我们已经对齐过）。
        """
        if fast:
            self._train_fast(corpus, vocab_size, verbose=verbose)
        else:
            self._train_naive(corpus, vocab_size, verbose=verbose)

    # ─────────────────────────────────────────
    # 2c. Naive 训练（O(n²) 量级）
    # ─────────────────────────────────────────
    def _train_naive(self, corpus: Corpus, vocab_size: int, verbose: bool = False) -> None:
        """
        最朴素的 BPE 训练：每一轮都重新扫描整个 word_freq 来统计 pair 频率。
        对小语料够用；对大语料慢，因为每轮 O(总 token 数)，总共 M 轮 → O(M·N)。

        【算法步骤】
            1) preprocess 得到 word_freq
            2) 初始 vocab = 所有出现过的单字符（包括 </w>）
            3) 循环直到 vocab 大小达到目标：
                a. get_stats 算出当前所有 pair 的频率
                b. 选频率最大的 pair（同频取字典序小的）
                c. merge_vocab 把这个 pair 合并到所有词里
                d. 把新 token 加进 vocab，pair 加进 merges
            4) is_trained = True
        """
        # 第 1 步：调统一接口里的 preprocess
        # 输入 ["low low low lowest newest"] 这样的句子列表
        # 输出 word_freq 形如 {"l o w </w>": 3, "l o w e s t </w>": 1, ...}
        word_freq = preprocess(corpus)

        # 第 2 步：初始化 vocab
        # 把每个词拆开，所有出现过的单字符都加进 vocab；</w> 也会自然包含
        self.vocab = set()
        for word in word_freq:
            for tok in word.split():
                self.vocab.add(tok)
        self.merges = []  # 清空（防止反复调用 train 时残留旧规则）

        if verbose:
            print(f"[naive init] vocab_size={len(self.vocab)}")

        # 第 3 步：主循环
        while len(self.vocab) < vocab_size:
            # 3a：统计当前所有 pair 频率
            stats = get_stats(word_freq)
            if not stats:
                # 已经没有任何相邻对了（所有词都被合并成单 token 了）
                break

            # 3b：选 best_pair
            # 这里用 min + (-count, pair) 的 trick：
            #   - 主键 -count：负数越小代表 count 越大 → min 自动选 count 最大的
            #   - 副键 pair  ：count 相同时取字典序**小**的
            # 为什么要取字典序小？因为 Fast 版本用 heapq 取最小，
            # 同频时也是取字典序小。两边规则一致，
            # assert naive.merges == fast.merges 才能过。
            best_pair = min(stats.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            best_freq = stats[best_pair]

            # 3c：把这个 pair 在所有词里合并
            word_freq = merge_vocab(best_pair, word_freq)

            # 3d：更新 vocab 和 merges
            new_token = best_pair[0] + best_pair[1]  # 字符串拼接，例如 ('l','o') → 'lo'
            self.vocab.add(new_token)
            self.merges.append(best_pair)

            if verbose:
                print(f"[naive merge {len(self.merges):>3}] "
                      f"{best_pair} (freq={best_freq}) -> {new_token!r}")

        # 第 4 步：标记已训练
        self.is_trained = True

    # ─────────────────────────────────────────
    # 2d. Fast 训练（heapq + 倒排索引 + 增量更新）
    # ─────────────────────────────────────────
    def _train_fast(self, corpus: Corpus, vocab_size: int, verbose: bool = False) -> None:
        """
        【核心思想】
            Naive 每轮要扫全语料统计 pair —— 浪费！因为合并一个 pair 只会影响
            "含这个 pair 的那几个词"，其它词的 pair 统计根本没变。
            Fast 版用三个数据结构维护"实时状态"，每轮只动受影响的部分：

            ─────────────────────────────────────────────────────────────
              words[i]     : 第 i 个词的当前 token 列表（list[str]，可变）
              freqs[i]     : 第 i 个词出现的次数（int）
              pair_freq[p] : pair p 现在在所有词里加权出现多少次（实时维护）
              pair_where[p]: 倒排索引！p 出现在哪些词里（set[word_idx]）
                             这样合并 p 时只要遍历 pair_where[p] 即可
              heap         : (-count, pair) 的最小堆 → pop 出来就是 count 最大的
                             允许"过期"条目（push 完不删旧的），pop 时校验
            ─────────────────────────────────────────────────────────────

        【为什么要用堆而不是每轮 max(pair_freq.items())】
            每轮 max() 是 O(P)（P 是 pair 种类数），M 轮总共 O(M·P)。
            堆 pop 是 O(log H)，push 也是 O(log H)。
            实际上由于每轮只更新少量 pair（受影响词产生的那些），
            push 的总数远小于全量重扫，所以总体复杂度更优。

        【堆的"过期条目"是怎么回事】
            假设 pair P 当前频率 5，堆里有 (-5, P)。
            后来某次合并让 P 的频率变成 3，我们 push (-3, P) 进堆。
            现在堆里同时有 (-5, P) 和 (-3, P) 两条记录。
            -5 比 -3 小（在 min-heap 里更靠前），所以会先 pop 出 (-5, P)。
            这是个"过期条目"，已经不反映真实频率了。
            我们的校验：pop 出来后比较 -neg 和 pair_freq[P] 是否相等。
            -5 != 3 → 过期 → 丢掉继续 pop，直到找到一条仍然匹配的为止。
        """

        # ── 准备：preprocess + 初始化所有数据结构 ──

        word_freq = preprocess(corpus)  # 和 Naive 一样的预处理

        # (1) 把 word_freq 拆成两个**并行数组** words 和 freqs
        # 为什么不直接用 dict？因为我们需要按 index 快速访问、修改每个词的 token list。
        # 用 list 索引比 dict 查找在循环里更方便。
        words: list[list[str]] = []
        freqs: list[int] = []
        self.vocab = set()
        for w, f in word_freq.items():
            toks = w.split()              # "l o w </w>" → ["l","o","w","</w>"]
            words.append(toks)
            freqs.append(f)
            self.vocab.update(toks)       # 顺手把单字符加进 vocab
        self.merges = []

        # (2) 一次性扫一遍，建好 pair_freq 和 pair_where
        # 这是整个训练里唯一一次"全扫" —— 之后全靠增量更新
        pair_freq: dict[tuple[str, str], int] = defaultdict(int)
        pair_where: dict[tuple[str, str], set[int]] = defaultdict(set)
        for wi, toks in enumerate(words):     # wi = word index（第几个词）
            f = freqs[wi]
            # zip(toks, toks[1:]) 是经典的"相邻对"写法
            # toks=[a,b,c,d] → zip 出 (a,b),(b,c),(c,d)
            for a, b in zip(toks, toks[1:]):
                pair_freq[(a, b)] += f         # 累加加权频次
                pair_where[(a, b)].add(wi)     # 记录这个词的下标

        # (3) 建堆：把 (-count, pair) 全推进去，heapify O(n)
        heap: list[tuple[int, tuple[str, str]]] = [(-c, p) for p, c in pair_freq.items()]
        heapq.heapify(heap)

        if verbose:
            print(f"[fast init] vocab_size={len(self.vocab)}, pairs={len(pair_freq)}")

        # ── 主循环 ──
        while len(self.vocab) < vocab_size:

            # ─── ① 从堆顶取出最高频 pair（跳过过期条目）───
            best_pair = None
            while heap:
                neg, p = heapq.heappop(heap)
                # 校验三件套：
                #   (a) -neg == pair_freq[p] : 频率值还是这个，没过期
                #   (b) -neg > 0             : 频率为正，pair 还存在
                # 如果是 0（一些 pair 被减成 0），那也算"过期"，跳过
                if -neg == pair_freq.get(p, 0) and -neg > 0:
                    best_pair = p
                    break
                # 否则就是过期条目，继续 pop 下一条
            if best_pair is None:
                # 堆空了，再没有可合并的 pair → 提前终止
                break

            best_freq = pair_freq[best_pair]
            new_tok = best_pair[0] + best_pair[1]   # 例如 ('l','o') → 'lo'
            self.vocab.add(new_tok)
            self.merges.append(best_pair)

            # ─── ② 只动含 best_pair 的词 ───
            # pair_where.pop 把这个 pair 的倒排表整个取出来（之后它不会再
            # 出现，因为已经被合并掉了）
            affected = pair_where.pop(best_pair, set())
            pair_freq.pop(best_pair, None)  # pair_freq 里也清掉，免得后面误判

            for wi in affected:
                f = freqs[wi]
                old_toks = words[wi]

                # 改前：用 Counter 统计这个词里**所有相邻对**的多重数
                # 例：old_toks = [A, A, A] → Counter 里 (A,A) 计 2 次
                # 这个多重数很重要：后面差分要乘以这个数
                old_counts = Counter(zip(old_toks, old_toks[1:]))

                # 执行合并：调用统一的 _apply_one_merge（和 tokenize 共用）
                # 这是为了**只写一份合并逻辑**，避免两处不一致
                new_toks = self._apply_one_merge(old_toks, best_pair)
                words[wi] = new_toks  # 原地替换

                # 改后：同样用 Counter 统计
                new_counts = Counter(zip(new_toks, new_toks[1:]))

                # ─── ③ 差分更新 pair_freq 和 pair_where ───
                # changed 集合 = "改前出现过" ∪ "改后出现过" 的所有 pair
                changed: set[tuple[str, str]] = set(old_counts) | set(new_counts)
                for p in changed:
                    # delta = (改后出现次数 - 改前出现次数) × 词频
                    # 注意要乘 f，因为一个词可能出现多次
                    delta = (new_counts.get(p, 0) - old_counts.get(p, 0)) * f
                    if delta != 0:
                        pair_freq[p] = pair_freq.get(p, 0) + delta

                    # 倒排索引也要更新：
                    #   - 如果改后这个词里还有 p 这个对 → 加进 pair_where[p]
                    #   - 如果改后没有了 → 从 pair_where[p] 里移掉
                    if new_counts.get(p, 0) > 0:
                        pair_where[p].add(wi)
                    else:
                        pair_where[p].discard(wi)

                    # ─── ④ 把新的频率值推进堆 ───
                    # 不需要去删旧条目（删除是 O(n)，不划算）
                    # 旧条目当成"过期"，靠 pop 时的校验过滤
                    cur = pair_freq.get(p, 0)
                    if cur > 0:  # 只推正频率的（频率为 0 没必要进堆）
                        heapq.heappush(heap, (-cur, p))

            if verbose:
                print(f"[fast merge {len(self.merges):>3}] "
                      f"{best_pair} (freq={best_freq}) -> {new_tok!r}")

        self.is_trained = True

    # ─────────────────────────────────────────
    # 2e. 推理：tokenize
    # ─────────────────────────────────────────
    def tokenize(self, text: str) -> TokenList:
        """
        把一段文本（可以是单词也可以是句子）切成 token 列表。

        ╭─── BPE 的歧义消解策略 ─────────────────────────────────────────╮
        │ BPE 推理**不是**贪心最长匹配（这点和 WordPiece 不一样！）。      │
        │ BPE 的标准做法：                                                │
        │   1) 把每个词拆成单字符（加上 </w> 词尾标记）                   │
        │   2) 按 self.merges 里记录的**顺序**，依次应用每条规则          │
        │                                                                 │
        │ 为什么这样能保证唯一性？                                        │
        │   想象 "newer" 这个词，理论上可以切成 ["new","er"] 或           │
        │   ["n","ewer"]。哪种对？如果用"贪心：哪个子词在 vocab 里就选哪个"│
        │   会依赖扫描方向和长度优先级，结果可能不唯一。                  │
        │   BPE 的回答是：merge 的**先后顺序**就是唯一的权威。            │
        │     - 训练完后 self.merges 是个固定的有序列表                   │
        │     - 推理时永远从单字符开始，严格按这个顺序应用                │
        │     - 每条规则的应用是确定的（左到右扫描，相邻匹配就合并）      │
        │   不管你"想象"过哪种切法，按规则顺序走下来结果总是同一个。      │
        │                                                                 │
        │   一句话总结：训练定顺序，推理走顺序，故唯一。                  │
        ╰─────────────────────────────────────────────────────────────────╯
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer 还没训练，请先调用 train()。")

        # 用和训练**同一套**预处理 —— 必须一致，否则 vocab 里的 token
        # 和现在生成的 token 不匹配，命中率会乱。
        word_freq = preprocess([text])
        # word_freq 的 key 已经是 "c h a r s </w>" 格式
        # 遍历每个词分别切分

        all_tokens: TokenList = []
        for char_word in word_freq:
            # 1) 拆成字符 list（已经是空格分隔的，直接 split）
            tokens = char_word.split()                # 例如 ["n","e","w","e","r","</w>"]

            # 2) 按 self.merges 顺序应用规则（核心步骤）
            tokens = self._apply_merges(tokens)

            # 3) 没在 vocab 里的子词 → [UNK]
            # 这种情况一般是字符级的 token：训练语料里没出现过这个字符
            # 比如训练只见过英文小写字母，推理时出现数字或大写就 UNK
            tokens = [t if t in self.vocab else "[UNK]" for t in tokens]

            all_tokens.extend(tokens)
        return all_tokens

    # ─────────────────────────────────────────
    # 2f. 单条 merge 规则的合并函数（原子操作）
    # ─────────────────────────────────────────
    @staticmethod                # 不依赖 self，做成 staticmethod 调用更清爽
    def _apply_one_merge(tokens: TokenList, pair: tuple[str, str]) -> TokenList:
        """
        把**一条** merge 规则 (A, B) 应用到 tokens 上：
        从左到右扫描，相邻 token 是 (A, B) 就合并成 "AB"，**不重叠**。

        "不重叠"的意思：
            tokens = [A, A, A] 应用 (A, A)：
                i=0: tokens[0]=A, tokens[1]=A → 合并成 "AA"，i 跳到 2
                i=2: 只剩 1 个 token，进 else 分支，append A，i=3
                结果：["AA", "A"]
            而不是 ["AA", "AA"]（那样会重叠）或 ["A","AA"]（那样是右优先）。
            BPE 标准就是从左到右非重叠。

        【为什么单独提出来一个函数】
            训练（_train_fast）和推理（tokenize）都需要"应用一条 merge 规则"。
            如果两边各写一份，万一行为不一致就会出 bug（比如训练用左到右、
            推理用右到左，结果会对不上）。所以提取出来共用，**只有一处定义**。
        """
        a, b = pair
        i = 0
        merged: TokenList = []
        n = len(tokens)
        while i < n:
            # 同时满足：还有下一个 token & 当前 == A & 下一个 == B
            if i < n - 1 and tokens[i] == a and tokens[i + 1] == b:
                merged.append(a + b)   # 字符串拼接得到新 token
                i += 2                 # 跳过 2 个 → 不重叠
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    # ─────────────────────────────────────────
    # 2g. 应用所有 merge 规则
    # ─────────────────────────────────────────
    def _apply_merges(self, tokens: TokenList) -> TokenList:
        """
        按 self.merges 里的顺序，对 tokens **依次**应用每条规则。
        每应用一条规则，tokens 都会被替换成合并后的新列表。
        """
        for pair in self.merges:
            tokens = self._apply_one_merge(tokens, pair)
        return tokens


# ─────────────────────────────────────────────────────────────────
# 3. 自测 / 小 demo
#    直接 `python bpe_cn.py` 就能跑，看每一块是否正常
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 块 1：get_stats 验证 ──
    word_freq = {"l o w </w>": 3, "n e w e s t </w>": 2}
    stats = get_stats(word_freq)

    print("输入 word_freq:")
    for k, v in word_freq.items():
        print(f"  {k!r}: {v}")

    print("\nget_stats 输出（按频次降序）:")
    for pair, freq in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {pair}: {freq}")

    # 期望（手算）：
    #   "l o w </w>"      (×3) 贡献 (l,o)=3, (o,w)=3, (w,</w>)=3
    #   "n e w e s t </w>"(×2) 贡献 (n,e)=2, (e,w)=2, (w,e)=2, (e,s)=2, (s,t)=2, (t,</w>)=2
    expected = {
        ("l", "o"): 3, ("o", "w"): 3, ("w", "</w>"): 3,
        ("n", "e"): 2, ("e", "w"): 2, ("w", "e"): 2,
        ("e", "s"): 2, ("s", "t"): 2, ("t", "</w>"): 2,
    }
    assert stats == expected, f"不一致!\n  got:      {stats}\n  expected: {expected}"
    print("\n✓ get_stats 与期望结果一致")

    # ── 块 2：Naive train() 验证 ──
    print("\n" + "=" * 60)
    print("块 2：Naive BPE 训练")
    print("=" * 60)
    corpus = ["low low low lowest newest"]
    bpe = BPETokenizer()
    # 这里特意用 fast=False，跑朴素版本，把每一步打印出来看清楚
    bpe.train(corpus, vocab_size=15, fast=False, verbose=True)
    print(f"\n最终 vocab ({len(bpe.vocab)}): {sorted(bpe.vocab)}")
    print(f"merges ({len(bpe.merges)}): {bpe.merges}")

    # ── 块 3：tokenize 验证 ──
    print("\n" + "=" * 60)
    print("块 3：tokenize（按 merges 顺序，解决歧义）")
    print("=" * 60)
    for w in ["lowest", "newest", "low", "newer", "unknown"]:
        # 'newer' 里有 'r'，训练语料里没见过 → 应该出现 [UNK]
        # 'unknown' 里 'u','k' 都没见过
        print(f"  tokenize({w!r:>10}) -> {bpe.tokenize(w)}")

    # 歧义消解的可视化演示：从三种不同的"起始切法"出发，跑 _apply_merges
    # 看结果对比。这能直观说明为什么 BPE 要求**从字符级开始**：
    # 只有字符级起点才是 canonical（标准、唯一）的。
    print("\n— 歧义消解演示：不同起始切法的最终结果 —")
    starts = {
        "char-split":       ["l", "o", "w", "e", "s", "t", "</w>"],
        "imagined-split-1": ["lo", "w", "e", "s", "t", "</w>"],   # 假装已经合好 "lo"
        "imagined-split-2": ["l", "ow", "est", "</w>"],            # 假装已经合好一部分
    }
    results = {}
    for name, toks in starts.items():
        out = bpe._apply_merges(toks)
        print(f"  {name:>18}: {toks}  ->  {out}")
        results[name] = out
    # 注意：非 canonical 的起点可能跳过依赖原始字符的早期规则，
    # 所以三个结果不一定相同。这正好说明 BPE 的规定："必须从字符开始"，
    # 这条规定 + 固定的 merge 顺序 → 唯一结果。
    assert bpe.tokenize("lowest") == bpe.tokenize("lowest"), "确定性失败"
    print("\n✓ 从字符级起点出发，tokenize 是确定且唯一的")

    # ── 块 4：Fast BPE 等价性 + benchmark ──
    print("\n" + "=" * 60)
    print("块 4：Fast BPE（heapq + 增量更新）")
    print("=" * 60)

    # (a) 小语料：Fast 和 Naive 必须产出完全一样的 vocab 和 merges
    # 这是回归测试的金标准：只要 assert 过了，就说明 Fast 没写错
    naive = BPETokenizer(); naive.train(corpus, vocab_size=15, fast=False)
    fast  = BPETokenizer(); fast.train(corpus,  vocab_size=15, fast=True)
    assert naive.vocab  == fast.vocab,  f"vocab 不一致:\n  naive={naive.vocab}\n  fast ={fast.vocab}"
    assert naive.merges == fast.merges, f"merges 不一致:\n  naive={naive.merges}\n  fast ={fast.merges}"
    print(f"✓ 小语料 (vocab_size=15)：naive.vocab == fast.vocab, naive.merges == fast.merges")

    # (b) 中等语料 benchmark
    # 小语料看不出速度差异（两边都是几毫秒），这里造一个 100 句、每句 20 词
    # 的合成语料，vocab_size 拉到 300，让 Naive 真的慢下来
    import random, time
    random.seed(42)  # 固定种子保证可复现
    alphabet = "abcdefghijklmnop"
    # 先随机造 80 个"基础词"，每个词 3-9 个字母
    base_words = ["".join(random.choice(alphabet) for _ in range(random.randint(3, 9)))
                  for _ in range(80)]
    # 然后从基础词里随机组合成 100 句话
    sentences = [" ".join(random.choice(base_words) for _ in range(20)) for _ in range(100)]
    medium_corpus = sentences
    target_vocab = 300

    # 计时：Naive
    t0 = time.perf_counter()
    n2 = BPETokenizer(); n2.train(medium_corpus, vocab_size=target_vocab, fast=False)
    t_naive = time.perf_counter() - t0

    # 计时：Fast
    t0 = time.perf_counter()
    f2 = BPETokenizer(); f2.train(medium_corpus, vocab_size=target_vocab, fast=True)
    t_fast = time.perf_counter() - t0

    # 再次断言：中等语料上两者结果也必须一致
    # （这是发现 merge_vocab bug 的关键测试！小语料上的 bug 没暴露）
    assert n2.vocab  == f2.vocab,  "中等语料 vocab 不一致"
    assert n2.merges == f2.merges, "中等语料 merges 不一致"
    speedup = t_naive / t_fast if t_fast > 0 else float("inf")
    print(f"\n  中等语料: {len(medium_corpus)} 句 × 20 词, target_vocab={target_vocab}")
    print(f"  Naive: {t_naive*1000:7.1f} ms")
    print(f"  Fast : {t_fast*1000:7.1f} ms")
    print(f"  ✓ vocab/merges 完全一致；Fast 加速比 ≈ {speedup:.1f}×")

    # (c) tokenize 在 Fast 训出来的模型上也能正常工作
    sample = "abcfgh"
    print(f"\n  fast.tokenize({sample!r}) -> {f2.tokenize(sample)}")
