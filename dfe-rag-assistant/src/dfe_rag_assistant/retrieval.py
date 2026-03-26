from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import AppConfig


_SYN_MAP = {
    "类型": ["形式", "种类", "分类"],
    "功率": ["额定功率", "容量", "出力"],
    "尺寸": ["长度", "规格", "参数"],
    "区别": ["差异", "对比", "不同点"],
    "基础": ["底座", "基础形式", "基础结构"],
    "材料": ["材质"],
}


def generate_query_variants(query: str, max_variants: int = 3) -> List[str]:
    """
    轻量 Query 理解：用规则做同义词扩展。
    输出包含原 query，并生成最多 max_variants 条变体。
    """
    query = query.strip()
    variants = [query]
    used = {query}

    for src, repls in _SYN_MAP.items():
        if src in query:
            for r in repls:
                cand = query.replace(src, r)
                if cand not in used:
                    variants.append(cand)
                    used.add(cand)
                    if len(variants) >= max_variants:
                        return variants
    return variants[:max_variants]


def _jiebacut(text: str) -> List[str]:
    import jieba

    return [t for t in jieba.lcut(text) if t.strip()]


@dataclass
class IndexBundle:
    faiss_index: Any
    bm25: Any
    chunks_meta: List[Dict[str, Any]]  # chunk_id 顺序不限，但应可通过 id 查
    id_to_idx: Dict[int, int]
    embedding_dim: int


def load_index(index_dir: str, cfg: AppConfig) -> IndexBundle:
    index_path = Path(index_dir)
    chunks_meta_path = index_path / cfg.paths.chunks_meta_json

    with open(chunks_meta_path, "r", encoding="utf-8") as f:
        chunks_meta = json.load(f)

    id_to_idx = {int(c["chunk_id"]): i for i, c in enumerate(chunks_meta)}

    # faiss index
    import faiss

    faiss_index = faiss.read_index(str(index_path / cfg.paths.faiss_index_path))

    # bm25
    with open(index_path / cfg.paths.bm25_path, "rb") as f:
        obj = pickle.load(f)
    bm25 = obj["bm25"]

    embedding_dim = int(faiss_index.d)
    return IndexBundle(
        faiss_index=faiss_index,
        bm25=bm25,
        chunks_meta=chunks_meta,
        id_to_idx=id_to_idx,
        embedding_dim=embedding_dim,
    )


def _rrf_fuse(
    vec_ranks: Dict[int, int],
    bm25_ranks: Dict[int, int],
    k: int,
) -> Dict[int, float]:
    """
    vec_ranks/bm25_ranks: chunk_id -> rank（1-based，越小越相关）
    """
    fused: Dict[int, float] = {}
    for cid, r in vec_ranks.items():
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + r)
    for cid, r in bm25_ranks.items():
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + r)
    return fused


def retrieve(
    query: str,
    bundle: IndexBundle,
    cfg: AppConfig,
    embedder: Any,
) -> List[Dict[str, Any]]:
    """
    多路召回 + RRF 融合，返回最终候选 chunk 列表（含 fusion_score）。
    """
    query_variants = generate_query_variants(query, max_variants=cfg.retrieval.max_query_variants)

    vec_ranks: Dict[int, int] = {}
    bm25_ranks: Dict[int, int] = {}

    # 1) Dense（FAISS）召回：对每个 query variant 都取 TopK，合并最小 rank
    for qv in query_variants:
        q_emb = embedder.encode([qv], normalize_embeddings=cfg.index.normalize_embeddings, show_progress_bar=False)
        q_emb = np.asarray(q_emb, dtype="float32")

        scores, ids = bundle.faiss_index.search(q_emb, cfg.retrieval.top_k_vec)
        # ids: shape (1, top_k)
        for rank0, (cid, _score) in enumerate(zip(ids[0].tolist(), scores[0].tolist())):
            if cid < 0:
                continue
            rank = rank0 + 1
            if cid not in vec_ranks or rank < vec_ranks[cid]:
                vec_ranks[cid] = rank

    # 2) BM25 召回：对每个 query variant 取 TopK，合并最小 rank
    for qv in query_variants:
        q_tokens = _jiebacut(qv)
        scores = bundle.bm25.get_scores(q_tokens)  # shape (n_chunks,)
        # top indices
        top_idx = np.argsort(-scores)[: cfg.retrieval.top_k_bm25]
        # 这里 bm25 的 corpus 顺序对应 chunk_id 是“构建时顺序”
        # 由于 build_index 里 chunk_meta 的 chunk_id 已经重映射全局唯一，
        # 我们通过 top_idx -> chunks_meta 查到对应 chunk_id。
        for rank0, idx in enumerate(top_idx.tolist()):
            cid = int(bundle.chunks_meta[idx]["chunk_id"])
            rank = rank0 + 1
            if cid not in bm25_ranks or rank < bm25_ranks[cid]:
                bm25_ranks[cid] = rank

    fused = _rrf_fuse(vec_ranks=vec_ranks, bm25_ranks=bm25_ranks, k=cfg.retrieval.rrf_k)

    # 3) 返回最终 TopK
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: cfg.retrieval.final_top_k]
    out: List[Dict[str, Any]] = []
    for cid, fs in ranked:
        meta_idx = bundle.id_to_idx.get(int(cid))
        if meta_idx is None:
            continue
        item = dict(bundle.chunks_meta[meta_idx])
        item["fusion_score"] = float(fs)
        item["vec_rank"] = int(vec_ranks.get(cid, 0)) if cid in vec_ranks else None
        item["bm25_rank"] = int(bm25_ranks.get(cid, 0)) if cid in bm25_ranks else None
        out.append(item)
    return out

