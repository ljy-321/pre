from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .chunking import chunk_elements
from .config import AppConfig, PathsConfig
from .pdf_utils import Element, extract_elements_from_pdf, iter_pdf_paths


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _jiebacut(text: str) -> List[str]:
    import jieba

    # 默认模式会把中文粗切；面试里你可以强调“BM25 对关键词更稳”
    return [t for t in jieba.lcut(text) if t.strip()]


def build_faiss_index(
    embeddings: np.ndarray,
) -> "Any":
    try:
        import faiss
    except Exception as e:  # pragma: no cover
        raise RuntimeError("缺少依赖 faiss-cpu，请先 pip install faiss-cpu") from e

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype("float32"))
    return index


def _save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_index(
    pdf_dir: str,
    index_dir: str,
    cfg: AppConfig,
    max_pages_per_pdf: Optional[int] = None,
    batch_size: int = 32,
) -> None:
    """
    从 PDF 构建：
    - FAISS 向量索引（chunk -> embedding）
    - BM25 稀疏索引（chunk -> jieba tokens）
    - chunks_meta.json（chunk_id -> content/来源）
    """
    pdf_paths = iter_pdf_paths(pdf_dir)
    if not pdf_paths:
        raise ValueError(f"pdf_dir 内没有找到 pdf：{pdf_dir}")

    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Dict[str, Any]] = []
    all_elements_meta: List[Dict[str, Any]] = []
    pdf_meta: List[Dict[str, Any]] = []

    # 1) PDF -> elements -> chunks
    for pdf_path in pdf_paths:
        elements = extract_elements_from_pdf(
            pdf_path,
            max_pages=max_pages_per_pdf,
        )
        all_elements_meta.extend(
            [{"pdf_path": pdf_path, **el.to_dict()} for el in elements]
        )
        pdf_meta.append({"pdf_path": pdf_path, "n_elements": len(elements)})

        chunks = chunk_elements(elements, cfg.chunking)
        # 给每个 chunk 记录来源 PDF（便于生成时 citation）
        for c in chunks:
            c["pdf_path"] = pdf_path
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("没有生成任何 chunk，请检查 PDF 解析/切块参数。")

    # chunk_id 在 chunk_elements 里是逐 pdf 从 0 开始的，这里需要重映射成全局唯一
    # 做法：重新编号并建立映射
    new_id = 0
    for c in all_chunks:
        c["global_chunk_id"] = new_id
        new_id += 1
    # 为了兼容后续代码，使用 global_chunk_id 作为最终 chunk_id
    for c in all_chunks:
        c["chunk_id"] = c["global_chunk_id"]
        del c["global_chunk_id"]

    # 2) embedding
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # pragma: no cover
        raise RuntimeError("缺少依赖 sentence-transformers，请先 pip install sentence-transformers") from e

    embedder = SentenceTransformer(cfg.index.embedding_model_name)

    texts = [c["content"] for c in all_chunks]
    embeddings_list: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch = texts[i : i + batch_size]
        emb = embedder.encode(batch, normalize_embeddings=False, show_progress_bar=False)
        embeddings_list.append(np.asarray(emb))

    embeddings = np.vstack(embeddings_list).astype("float32")
    if cfg.index.normalize_embeddings:
        embeddings = _normalize_rows(embeddings)

    # 3) FAISS 索引
    index = build_faiss_index(embeddings)
    import faiss

    faiss.write_index(index, str(index_path / cfg.paths.faiss_index_path))

    # 4) BM25 索引
    from rank_bm25 import BM25Okapi

    corpus_tokens = []
    for c in tqdm(all_chunks, desc="BM25 tokenizing"):
        corpus_tokens.append(_jiebacut(c["content"]))

    bm25 = BM25Okapi(corpus_tokens)

    with open(index_path / cfg.paths.bm25_path, "wb") as f:
        pickle.dump(
            {"bm25": bm25, "chunk_ids": [c["chunk_id"] for c in all_chunks]},
            f,
        )

    # 5) 元数据
    _save_json(index_path / cfg.paths.chunks_meta_json, all_chunks)
    _save_json(index_path / cfg.paths.elements_meta_json, all_elements_meta[:2000])  # 防止过大
    _save_json(index_path / cfg.paths.pdf_meta_json, pdf_meta)

    # 保存配置快照（用于复现）
    _save_json(index_path / "run_config.json", asdict(cfg))

    print(f"[OK] 索引构建完成：chunks={len(all_chunks)} FAISS-> {cfg.paths.faiss_index_path}")

