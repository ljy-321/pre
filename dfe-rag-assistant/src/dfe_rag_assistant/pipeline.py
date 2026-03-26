from __future__ import annotations

from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer

from .config import AppConfig
from .generation import RagGenerator
from .indexing import build_index
from .retrieval import IndexBundle, load_index, retrieve
from .reranking import maybe_rerank


def make_embedder(cfg: AppConfig) -> SentenceTransformer:
    return SentenceTransformer(cfg.index.embedding_model_name)


def chat(
    query: str,
    index_dir: str,
    cfg: AppConfig,
) -> Dict[str, Any]:
    bundle = load_index(index_dir, cfg)
    embedder = make_embedder(cfg)

    candidates = retrieve(query=query, bundle=bundle, cfg=cfg, embedder=embedder)
    reranked = maybe_rerank(query=query, candidates=candidates, cfg=cfg)

    gen = RagGenerator(cfg)
    result = gen.answer(query=query, candidates=reranked)

    return {
        "query": query,
        "answer": result.answer,
        "confidence": result.confidence,
        "sources": result.sources,
        "retrieved_chunks": [
            {
                "chunk_id": c.get("chunk_id"),
                "pdf_path": c.get("pdf_path"),
                "page_num": c.get("page_num"),
                "fusion_score": c.get("fusion_score"),
                "rerank_score": c.get("rerank_score"),
                "content_preview": (c.get("content") or "")[:200],
            }
            for c in reranked[: cfg.retrieval.final_top_k]
        ],
    }


def build_index_cli(
    pdf_dir: str,
    index_dir: str,
    cfg: AppConfig,
    max_pages_per_pdf: int | None = None,
) -> None:
    build_index(
        pdf_dir=pdf_dir,
        index_dir=index_dir,
        cfg=cfg,
        max_pages_per_pdf=max_pages_per_pdf,
    )

