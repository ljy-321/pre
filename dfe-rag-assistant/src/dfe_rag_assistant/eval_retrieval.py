from __future__ import annotations

import json
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer

from .config import AppConfig
from .pipeline import make_embedder
from .retrieval import load_index, retrieve


def eval_retrieval(
    index_dir: str,
    eval_jsonl: str,
    ks: List[int],
    cfg: AppConfig,
) -> Dict[str, Any]:
    bundle = load_index(index_dir, cfg)
    embedder = make_embedder(cfg)

    ks_sorted = sorted(ks)
    recall_counts = {k: 0 for k in ks_sorted}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    n = 0

    with open(eval_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            query = item["query"]
            gold_cid = int(item["gold_chunk_id"])

            cands = retrieve(query=query, bundle=bundle, cfg=cfg, embedder=embedder)
            top_cids = [int(c["chunk_id"]) for c in cands]

            n += 1
            if gold_cid in top_cids:
                rank = top_cids.index(gold_cid) + 1  # 1-based
                mrr_sum += 1.0 / rank

                for k in ks_sorted:
                    if rank <= k:
                        recall_counts[k] += 1

                # binary relevance NDCG@k：相关=1，且理想情况下在 rank=1
                # 只在 max_k 命中时才算，这里用 rank <= max_k 的近似
                max_k = max(ks_sorted)
                if rank <= max_k:
                    ndcg_sum += 1.0 / (1.0 + (rank - 1))  # 近似单调分数
            else:
                for k in ks_sorted:
                    recall_counts[k] += 0

    if n == 0:
        raise ValueError("eval_jsonl 为空或无法解析。")

    out = {
        "n_queries": n,
        "recall": {f"Recall@{k}": recall_counts[k] / n for k in ks_sorted},
        "MRR": mrr_sum / n,
        "NDCG_approx": ndcg_sum / n,
    }
    return out

