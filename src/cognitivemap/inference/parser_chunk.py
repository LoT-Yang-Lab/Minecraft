"""
序列分块（chunk parsing）工具。

这个模块的职责很单一：
给定一条原始 token 序列，以及一组候选 chunk 名称（例如 "C1-C2"、"C1-C2-M12"），
把原始序列重新解析成“尽可能长的 chunk的序列，即使用chunk尽可能少”。

当前实现采用 Trie + 从左到右最长匹配：
1. 先把所有 chunk 按 "-" 拆成 token 列表并插入 Trie。
2. 从序列第 0 个位置开始，尝试找到当前位置能命中的最长 chunk。
3. 如果没有任何 chunk 命中，则退回到单 token。
4. 重复直到扫完整条序列。

这套逻辑会被 miner_chunk.py 调用：
每当发现一批新的候选模式后，就用这里的解析器在“原始序列”上重新切分一次，
从而得到更高层、更紧凑的 chunk_sequence。
"""


class TrieNode:
    """Trie 的单个节点，用于表示 chunk token 前缀。"""

    def __init__(self):
        """初始化子节点映射和 chunk 结束标记。"""

        # children: 当前节点沿着不同 token 往下走的分支。
        # 例如已经插入 ["C1", "C2"] 和 ["C1", "C3"] 后，
        # 根节点的 children 里会有 "C1"，而 "C1" 节点下面会继续有 "C2"、"C3"。
        self.children = {}
        # is_end 表示“从根走到当前节点”这条路径是否刚好对应一个完整 chunk。
        # 它允许更长 chunk 和更短 chunk 共存，例如：
        # - "C1-C2"
        # - "C1-C2-M12"
        self.is_end = False


class TokenTrie:
    """支持 chunk 最长匹配的 token 前缀树。"""

    def __init__(self):
        """创建空 Trie 根节点。"""

        self.root = TrieNode()

    def insert(self, tokens):
        """
        把一个 chunk 对应的 token 列表插入 Trie。

        参数
        ----
        tokens:
            一个 chunk 拆开后的 token 列表，例如 ['C3', 'M13']。
        """
        node = self.root
        for tok in tokens:
            if tok not in node.children:
                node.children[tok] = TrieNode()
            node = node.children[tok]
        # 走完整条 token 路径后，把最后一个节点标记为“一个合法 chunk 的结束位置”。
        node.is_end = True

    def match_longest(self, seq, start):
        """
        从 seq[start] 开始，返回能够匹配到的最长 chunk。

        例如：
        - seq = ["C1", "C2", "M12", "C3"]
        - Trie 中同时有 "C1-C2" 和 "C1-C2-M12"
        - start = 0
        则返回 ["C1", "C2", "M12"]，而不是更短的 ["C1", "C2"]。

        返回值为 token 列表；若当前位置无法命中任何 chunk，则返回 None。
        """
        node = self.root
        # longest 记录“到目前为止最近一次命中的完整 chunk”。
        # 之所以不是一命中就立刻返回，是因为后面可能还能继续向下匹配出更长的 chunk。
        longest = None
        # matched 用来累计从 start 开始一路匹配到的 token 路径。
        matched = []

        for i in range(start, len(seq)):
            tok = seq[i]
            if tok not in node.children:
                # 一旦当前 token 无法继续沿 Trie 往下走，
                # 就说明更长的 chunk 不存在了，循环结束。
                break
            node = node.children[tok]
            matched.append(tok)
            if node.is_end:
                # 只要当前路径对应一个完整 chunk，就更新 longest。
                # 如果后面还能继续匹配，longest 还可能被更长的 chunk 覆盖。
                longest = list(matched)
        return longest


def parse_with_chunks(seq, chunks, need_index=False):
    """
    使用给定 chunk 词表对原始序列做一次“最长匹配”解析。

    参数
    ----
    seq:
        原始 token 序列，例如 ["C1", "C2", "M12", "C3"]。
    chunks:
        候选 chunk 名称列表，元素是用 "-" 连接的字符串，例如：
        ["C1", "C2", "C1-C2", "C1-C2-M12"]。
        注意这里通常既包含原子 token，也包含多 token chunk。
    need_index:
        是否额外返回每个解析结果在原始序列中的起始下标。

    返回
    ----
    如果 need_index=False:
        返回解析后的 chunk 序列，例如 ["C1-C2-M12", "C3"]。
    如果 need_index=True:
        返回 (解析结果, 起始下标列表)。
    """

    def chunk_to_tokens(chunk):
        """按项目约定把 chunk 名称拆回 primitive token 列表。"""

        # 当前项目约定 chunk 名称内部使用 "-" 连接多个 token。
        return chunk.split("-")

    trie = TokenTrie()

    # 先把词表中的所有 chunk 全部灌进 Trie，后续解析时就可以复用同一棵前缀树。
    for ck in chunks:
        tokens = chunk_to_tokens(ck)
        trie.insert(tokens)

    # 从左到右扫描原始序列。
    # 这里的策略是“局部最长匹配”：
    # 每次在当前位置尽量吃掉最长的 chunk，然后跳到 chunk 末尾继续。
    res = []
    # indices[i] 表示 res[i] 这个 chunk 在原始 seq 中的起始位置。
    indices = []
    i = 0
    while i < len(seq):
        match = trie.match_longest(seq, i)
        if match is None:
            # fallback：当前位置没有任何 chunk 可以匹配时，退回成单 token。
            # 这样可以保证解析过程永远不会卡住。
            indices.append(i)
            res.append(seq[i])
            i += 1
        else:
            indices.append(i)
            # match 内部是 token 列表；输出时重新拼回统一的 chunk 命名格式。
            res.append("-".join(match))
            # 既然已经消费掉 len(match) 个 token，就整体跳过它们。
            i += len(match)
    if need_index:
        return res, indices
    return res
