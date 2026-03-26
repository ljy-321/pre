from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChunkingConfig:
    # 以“字符”为粗粒度单位（适配中文文档的面试讲法与实验参数）
    chunk_size_chars: int = 512
    overlap_chars: int = 100
    min_chunk_chars: int = 80


@dataclass
class RetrievalConfig:
    top_k_vec: int = 50
    top_k_bm25: int = 50
    final_top_k: int = 5

    # RRF 参数：k 越大，越不敏感于名次；k=60 是常用起点
    rrf_k: int = 60

    # query 扩展的变体数量上限
    max_query_variants: int = 3


@dataclass
class IndexConfig:
    embedding_model_name: str = "BAAI/bge-large-zh-v1.5"
    faiss_index_factory: str = "Flat"  # 当前实现：Flat + cosine（可扩展）

    # 归一化后用内积=cosine 相似度
    normalize_embeddings: bool = True


@dataclass
class GenerationConfig:
    # 如果 vLLM 可用，会优先使用 vLLM
    llm_backend: str = "transformers"  # "vllm" 或 "transformers"
    llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    # 拒答阈值：使用“重排/融合得分”做置信代理
    confidence_threshold: float = 0.3

    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.1


@dataclass
class PathsConfig:
    index_dir: str = "data/index_dfe"
    index_name: str = "dfe_rag"

    pdf_meta_json: str = "pdf_meta.json"
    elements_meta_json: str = "elements_meta.json"
    chunks_meta_json: str = "chunks_meta.json"

    faiss_index_path: str = "faiss.index"
    bm25_path: str = "bm25.pkl"

    # reranker（可选）
    rerank_enabled: bool = False
    rerank_model_name: str = "BAAI/bge-reranker-large"

    # 运行时缓存：让多次 chat 更快
    cache_path: Optional[str] = None


@dataclass
class AppConfig:
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    index: IndexConfig = IndexConfig()
    generation: GenerationConfig = GenerationConfig()
    paths: PathsConfig = PathsConfig()

