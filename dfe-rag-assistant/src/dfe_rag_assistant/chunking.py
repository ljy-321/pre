from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from .config import ChunkingConfig
from .pdf_utils import Element


_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s*|\n+")


def _normalize_newlines(text: str) -> str:
    # 让换行更适合分句
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text_to_chunks(
    text: str,
    cfg: ChunkingConfig,
) -> List[str]:
    """
    以“句子”为单位打包，chunk_size/overlap 用字符数近似控制。
    """
    text = _normalize_newlines(text)
    if not text:
        return []

    # 先按句号/问号/感叹号或换行切成句子
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if not sents:
        return [text[: cfg.chunk_size_chars]]

    # 计算 prefix（句子之间拼接时额外 +1 作为分隔）
    lens = [len(s) for s in sents]
    prefix = [0]
    for i, L in enumerate(lens):
        prefix.append(prefix[-1] + L + 1)

    def span_len(i: int, j: int) -> int:
        # i..j inclusive
        return prefix[j + 1] - prefix[i]

    chunks: List[str] = []
    i = 0
    n = len(sents)
    while i < n:
        # 找到最大 j，使得长度不超过 chunk_size
        j = i
        while j + 1 < n and span_len(i, j + 1) <= cfg.chunk_size_chars:
            j += 1

        chunk = "\n".join(sents[i : j + 1]).strip()
        if len(chunk) >= cfg.min_chunk_chars:
            chunks.append(chunk)
        else:
            # 太短的块：直接并到下一个块（通过跳过 i 实现“吸收”效果）
            pass

        if j >= n - 1:
            break

        # 下一块起点：保证与上一块末尾重叠 overlap_chars
        # 找最小 k，使得 k..j 的长度 >= overlap_chars
        k = j
        while k > i and span_len(k, j) < cfg.overlap_chars:
            k -= 1
        # 防止死循环
        next_i = max(k, i + 1)
        i = next_i

    return chunks


def chunk_elements(
    elements: List[Element],
    cfg: ChunkingConfig,
) -> List[Dict[str, Any]]:
    """
    表格（element_type="table"）作为原子块，不再常规切分。
    文本使用 split_text_to_chunks。
    """
    chunks: List[Dict[str, Any]] = []
    chunk_id = 0
    for el in elements:
        if el.element_type == "table":
            content = el.content.strip()
            if len(content) >= cfg.min_chunk_chars:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "content": content,
                        "page_num": el.page_num,
                        "element_type": "table",
                        "col_id": el.col_id,
                    }
                )
                chunk_id += 1
            continue

        # text
        for c in split_text_to_chunks(el.content, cfg):
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "content": c,
                    "page_num": el.page_num,
                    "element_type": "text",
                    "col_id": el.col_id,
                }
            )
            chunk_id += 1

    return chunks

