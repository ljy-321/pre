from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .config import AppConfig
from .pipeline import build_index_cli, chat
from .eval_retrieval import eval_retrieval


def _make_cfg_from_args(args: argparse.Namespace) -> AppConfig:
    cfg = AppConfig()
    cfg.chunking.chunk_size_chars = args.chunk_size_chars
    cfg.chunking.overlap_chars = args.overlap_chars
    cfg.index.embedding_model_name = args.embedding_model
    cfg.retrieval.final_top_k = args.final_top_k
    cfg.generation.llm_backend = args.llm_backend
    cfg.generation.llm_model_name = args.llm_model
    cfg.paths.index_dir = args.index_dir
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(prog="dfe-rag-assistant", description="东方电气智能培训助手（RAG）脚手架")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # build-index
    p_build = sub.add_parser("build-index", help="构建向量索引 + BM25 索引")
    p_build.add_argument("--pdf_dir", type=str, required=True, help="PDF 所在目录")
    p_build.add_argument("--index_dir", type=str, required=True, help="索引输出目录")
    p_build.add_argument("--chunk_size_chars", type=int, default=512)
    p_build.add_argument("--overlap_chars", type=int, default=100)
    p_build.add_argument("--embedding_model", type=str, default="BAAI/bge-large-zh-v1.5")
    p_build.add_argument("--final_top_k", type=int, default=5)
    p_build.add_argument("--max_pages_per_pdf", type=int, default=0)

    # chat
    p_chat = sub.add_parser("chat", help="基于 RAG 检索 + 生成回答")
    p_chat.add_argument("--index_dir", type=str, required=True)
    p_chat.add_argument("--query", type=str, required=True)
    p_chat.add_argument("--llm_backend", type=str, default="transformers", choices=["transformers", "vllm"])
    p_chat.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p_chat.add_argument("--final_top_k", type=int, default=5)

    # eval
    p_eval = sub.add_parser("eval", help="评测 Recall@K（offline）")
    p_eval.add_argument("--index_dir", type=str, required=True)
    p_eval.add_argument("--eval_jsonl", type=str, required=True, help="jsonl: {query, gold_chunk_id}")
    p_eval.add_argument("--ks", type=str, default="3,5,10")

    args = parser.parse_args()

    if args.cmd == "build-index":
        cfg = _make_cfg_from_args(args)
        max_pages = None if args.max_pages_per_pdf <= 0 else args.max_pages_per_pdf
        build_index_cli(pdf_dir=args.pdf_dir, index_dir=args.index_dir, cfg=cfg, max_pages_per_pdf=max_pages)
        return

    if args.cmd == "chat":
        cfg = _make_cfg_from_args(args)
        out = chat(query=args.query, index_dir=args.index_dir, cfg=cfg)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.cmd == "eval":
        ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
        cfg = AppConfig()
        out = eval_retrieval(index_dir=args.index_dir, eval_jsonl=args.eval_jsonl, ks=ks, cfg=cfg)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()

