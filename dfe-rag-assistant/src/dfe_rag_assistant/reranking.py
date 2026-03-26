from __future__ import annotations

from typing import Any, Dict, List

from .config import AppConfig


def maybe_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    cfg: AppConfig,
) -> List[Dict[str, Any]]:
    """
    可选重排：优先使用 FlagEmbedding（Cross-Encoder）。
    如果依赖不可用，退化到按 fusion_score 排序。
    """
    if not cfg.paths.rerank_enabled:
        return sorted(candidates, key=lambda x: float(x.get("fusion_score", 0.0)), reverse=True)

    try:
        from FlagEmbedding import FlagReranker
    except Exception:
        # 依赖缺失则退化
        return sorted(candidates, key=lambda x: float(x.get("fusion_score", 0.0)), reverse=True)

    try:
        reranker = FlagReranker(cfg.paths.rerank_model_name, use_fp16=True)
        pairs = [(query, c["content"]) for c in candidates]
        # 不同版本 compute_score 输入格式可能不同，这里做一个兼容尝试
        try:
            scores = reranker.compute_score(pairs)
        except Exception:
            scores = reranker.compute_score([[p[0], p[1]] for p in pairs])

        scored = []
        for c, s in zip(candidates, scores):
            item = dict(c)
            item["rerank_score"] = float(s)
            scored.append(item)
        return sorted(scored, key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
    except Exception:
        # 任何失败都不影响主链路
        return sorted(candidates, key=lambda x: float(x.get("fusion_score", 0.0)), reverse=True)

