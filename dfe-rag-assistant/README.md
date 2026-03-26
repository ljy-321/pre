# 东方电气智能培训助手（RAG 备战实习版）

这是一个“面试可讲清、Demo 可跑、结构可扩展”的企业知识问答系统（RAG）项目脚手架。它按你给的“切块 -> 索引 -> 多路召回 -> Rerank -> 生成 -> 引用与拒答 -> 评测闭环”的主线实现，并把你需要背的关键点（为什么这么做、怎么验证、怎么讲贡献）写进文档。

> 说明：由于不同同学的运行环境不同（Windows 是否能跑 vLLM、GPU/内存差异等），生成模块提供了 `vLLM`（如可用）和 `Transformers`（回退）两条路径；检索模块同样提供可选重排（如装了 `FlagEmbedding`）。

## 1. 你要理解的基础概念（面试常问）

### 1.1 RAG 到底在做什么？
RAG（Retrieval-Augmented Generation）把“搜索”与“生成”结合起来：
1. **检索（Retrieval）**：从企业文档库中找到最相关的片段（chunk）
2. **生成（Generation）**：把检索到的片段喂给大模型生成回答
3. **引用/拒答（Grounding）**：要求回答必须基于证据；证据不足则拒答，避免胡编

### 1.2 chunk 是什么？为什么要做切块？
企业 PDF 里的信息通常很长，不能直接整篇丢给模型。我们把文档拆成小片段（chunk），并为每个 chunk 建索引。chunk 大小时常见权衡：
- 太小：语义碎片化，召回不足（语义不完整）
- 太大：噪声增大、超上下文、精确匹配下降（答非所问）

### 1.3 向量召回 vs BM25
- **向量召回（Dense）**：用 embedding 表示语义相似度，擅长“同义表达的语义匹配”，但对精确关键词可能不稳。
- **BM25（Sparse）**：基于关键词（TF-IDF + 长度归一），对“型号/参数/精确术语”更稳，但对同义词泛化弱。
- **混合召回（Hybrid）**：用互补性提升整体 Recall。

### 1.4 RRF 融合为什么有效？
RRF（Reciprocal Rank Fusion）不依赖分数标定，直接利用各路召回的“排序名次”：
`score(d) = Σ 1 / (k + rank_d)`
这对不同模型（向量、BM25）的分数尺度差异很友好。

### 1.5 Reranker（重排）在解决什么？
检索召回阶段通常会取较多候选（如 Top20），再用更“懂匹配”的模型做二次排序，把最相关的 chunk 提到前面。常见做法是：
- 召回：快（Dense/BM25）
- 重排：准（Cross-Encoder/交叉编码器）

重排越准，生成端可用的证据越可靠，从而减少幻觉。

### 1.6 幻觉为什么会发生？怎么缓解？
常见原因是：模型没有看到足够证据，仍然“为了回答而回答”。缓解策略通常是组合拳：
- **证据绑定**：Prompt 强制“只基于 context”
- **阈值拒答**：召回置信度低则拒答
- **生成后引用校验（进阶）**：检查答案是否能对应到证据片段中的事实

## 2. 项目主线与 4 周里程碑（按时间做什么）

你可以把下面的内容当作“面试现场时间线叙述模板”。建议每周都交付一个“可量化结果”。

### 第 0-2 天：确定目标与评测体系
1. 明确训练/评测目标：Recall（召回正确）-> Accuracy（生成正确）-> 拒答率（证据不足不胡编）
2. 搭建最小基线（只做切块 + 向量索引 + TopK 检索）
3. 选定 chunk_size / overlap 的候选范围（如你给的 256/512/1024 + 50/100/200）

### 第 1 周：数据处理 + 基线搭建（切块与索引）
1. PDF 解析（多栏排序 + 表格保护）
2. 切块实验：对比 chunk_size/overlap（记录 chunk 数量、平均长度、Recall@5）
3. 输出索引：向量索引（FAISS）+ BM25 索引（rank_bm25）
4. 产出 Demo：能回答“最常见事实型问题”，并展示引用片段

对应重要代码（从文件名可直接在项目里定位）：为便于阅读，README 里只截取关键片段（有的地方用 `...` 表示省略），你面试讲的时候可以直接对照仓库里的完整实现文件。

- `src/dfe_rag_assistant/pdf_utils.py`（来自 `extract_elements_from_pdf` / `Element`：多栏排序 + 表格保护作为原子证据块）

```python
# 来自：src/dfe_rag_assistant/pdf_utils.py
@dataclass
class Element:
    element_type: str  # "text" | "table"
    content: str
    page_num: int
    bbox: Optional[List[float]] = None
    col_id: int = 0

def extract_elements_from_pdf(
    pdf_path: str,
    max_pages: Optional[int] = None,
    assume_two_columns: bool = True,
) -> List[Element]:
    blocks = page.get_text("blocks")
    for b in blocks:
        x0, y0, x1, y1, text, _block_no, _block_type = b
        cleaned = _clean_text_block(str(text))
        if assume_two_columns:
            x_center = (float(x0) + float(x1)) / 2.0
            col_id = 0 if x_center < page_w / 2.0 else 1
        text_blocks.append(
            Element(element_type="text", content=cleaned, page_num=page_idx, bbox=[x0, y0, x1, y1], col_id=col_id)
        )
    text_blocks.sort(key=lambda e: (e.col_id, e.bbox[1] if e.bbox else 0))

    tables = page.extract_tables()
    for t in tables or []:
        md = _table_to_markdown(t)
        if not md:
            continue
        elements.append(
            Element(element_type="table", content=md, page_num=page_idx, bbox=None, col_id=0)
        )
    return elements
```

- `src/dfe_rag_assistant/chunking.py`（来自 `split_text_to_chunks` / `chunk_elements`：chunk_size/overlap + table 原子块不切碎）

```python
# 来自：src/dfe_rag_assistant/chunking.py
def split_text_to_chunks(text: str, cfg: ChunkingConfig) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    while i < n:
        j = i
        while j + 1 < n and span_len(i, j + 1) <= cfg.chunk_size_chars:
            j += 1
        chunk = "\n".join(sents[i : j + 1]).strip()
        if len(chunk) >= cfg.min_chunk_chars:
            chunks.append(chunk)

        k = j
        while k > i and span_len(k, j) < cfg.overlap_chars:
            k -= 1
        i = max(k, i + 1)
    return chunks

def chunk_elements(elements: List[Element], cfg: ChunkingConfig) -> List[Dict[str, Any]]:
    for el in elements:
        if el.element_type == "table":
            chunks.append({
                "chunk_id": chunk_id,
                "content": el.content.strip(),
                "page_num": el.page_num,
                "element_type": "table",
                "col_id": el.col_id,
            })
            chunk_id += 1
            continue

        for c in split_text_to_chunks(el.content, cfg):
            chunks.append({
                "chunk_id": chunk_id,
                "content": c,
                "page_num": el.page_num,
                "element_type": "text",
                "col_id": el.col_id,
            })
            chunk_id += 1
    return chunks
```

- `src/dfe_rag_assistant/indexing.py`（来自 `build_index`：FAISS 向量索引 + BM25 稀疏索引 + 元数据落盘）

```python
# 来自：src/dfe_rag_assistant/indexing.py
for pdf_path in pdf_paths:
    elements = extract_elements_from_pdf(pdf_path, max_pages=max_pages_per_pdf)
    chunks = chunk_elements(elements, cfg.chunking)
    for c in chunks:
        c["pdf_path"] = pdf_path
    all_chunks.extend(chunks)

embedder = SentenceTransformer(cfg.index.embedding_model_name)
embeddings = embedder.encode(texts, normalize_embeddings=False, show_progress_bar=False)
embeddings = _normalize_rows(embeddings) if cfg.index.normalize_embeddings else embeddings

index = faiss.IndexFlatIP(d)  # 归一化后用内积≈余弦相似度
index.add(embeddings.astype("float32"))
faiss.write_index(index, str(index_path / cfg.paths.faiss_index_path))

from rank_bm25 import BM25Okapi
corpus_tokens = [_jiebacut(c["content"]) for c in all_chunks]
bm25 = BM25Okapi(corpus_tokens)
pickle.dump({"bm25": bm25, "chunk_ids": [c["chunk_id"] for c in all_chunks]}, f)

_save_json(index_path / cfg.paths.chunks_meta_json, all_chunks)
_save_json(index_path / "run_config.json", asdict(cfg))  # 用于复现你当时的 chunk/embedding 配置
```

- `src/dfe_rag_assistant/pipeline.py` + `demo_gradio.py`（来自 `chat` / `_run`：Demo 展示 `answer` 和可追溯的 `sources`）

```python
# 来自：src/dfe_rag_assistant/pipeline.py -> chat
bundle = load_index(index_dir, cfg)
embedder = make_embedder(cfg)
candidates = retrieve(query=query, bundle=bundle, cfg=cfg, embedder=embedder)
reranked = maybe_rerank(query=query, candidates=candidates, cfg=cfg)
gen = RagGenerator(cfg)
result = gen.answer(query=query, candidates=reranked)

return {
    "answer": result.answer,
    "sources": result.sources,  # 引用证据定位
    "retrieved_chunks": [
        {"chunk_id": c.get("chunk_id"), "pdf_path": c.get("pdf_path"), "page_num": c.get("page_num"), ...}
        for c in reranked[: cfg.retrieval.final_top_k]
    ],
}

# 来自：demo_gradio.py -> _run
out = chat(query=query, index_dir=index_dir, cfg=cfg)
return out["answer"], out["sources"]
```

### 第 2 周：Query 理解 + 多路召回 + 召回评测
1. 实现轻量 Query 扩展（同义词规则 + BM25 分词天然覆盖）
2. 实现 3 路召回（向量/BM25/融合 RRF）
3. 自建评测集（query -> gold chunk/doc）
4. 产出评测表：Recall@3/5/10，按问题类型（事实/对比/汇总）分桶分析

对应重要代码：为便于阅读，README 里只截取关键片段（有的地方用 `...` 表示省略），你面试讲的时候可以直接对照仓库里的完整实现文件。

- `src/dfe_rag_assistant/retrieval.py`（来自 `generate_query_variants` / `retrieve`：轻量同义词 + Dense/BM25 多路召回 + RRF 融合）

```python
# 来自：src/dfe_rag_assistant/retrieval.py -> generate_query_variants
_SYN_MAP = {
    "类型": ["形式", "种类", "分类"],
    "功率": ["额定功率", "容量", "出力"],
    "尺寸": ["长度", "规格", "参数"],
    "区别": ["差异", "对比", "不同点"],
}

def generate_query_variants(query: str, max_variants: int = 3) -> List[str]:
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

# 来自：src/dfe_rag_assistant/retrieval.py -> RRF 融合与 retrieve 主流程
def _rrf_fuse(vec_ranks: Dict[int, int], bm25_ranks: Dict[int, int], k: int) -> Dict[int, float]:
    fused = {}
    for cid, r in vec_ranks.items():
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + r)
    for cid, r in bm25_ranks.items():
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + r)
    return fused

def retrieve(query: str, bundle: IndexBundle, cfg: AppConfig, embedder: Any) -> List[Dict[str, Any]]:
    query_variants = generate_query_variants(query, max_variants=cfg.retrieval.max_query_variants)
    vec_ranks, bm25_ranks = {}, {}

    # Dense（FAISS）
    for qv in query_variants:
        q_emb = embedder.encode([qv], normalize_embeddings=cfg.index.normalize_embeddings, show_progress_bar=False)
        scores, ids = bundle.faiss_index.search(q_emb, cfg.retrieval.top_k_vec)
        for rank0, (cid, _score) in enumerate(zip(ids[0].tolist(), scores[0].tolist())):
            rank = rank0 + 1
            if cid not in vec_ranks or rank < vec_ranks[cid]:
                vec_ranks[cid] = rank

    # BM25
    for qv in query_variants:
        q_tokens = _jiebacut(qv)
        scores = bundle.bm25.get_scores(q_tokens)
        top_idx = np.argsort(-scores)[: cfg.retrieval.top_k_bm25]
        for rank0, idx in enumerate(top_idx.tolist()):
            cid = int(bundle.chunks_meta[idx]["chunk_id"])
            rank = rank0 + 1
            if cid not in bm25_ranks or rank < bm25_ranks[cid]:
                bm25_ranks[cid] = rank

    fused = _rrf_fuse(vec_ranks=vec_ranks, bm25_ranks=bm25_ranks, k=cfg.retrieval.rrf_k)
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: cfg.retrieval.final_top_k]
    return [ ... 带 fusion_score 的 chunk meta ... ]
```

- `src/dfe_rag_assistant/eval_retrieval.py`（来自 `eval_retrieval`：离线计算 Recall@K / MRR / NDCG）

```python
# 来自：src/dfe_rag_assistant/eval_retrieval.py
recall_counts = {k: 0 for k in ks_sorted}
mrr_sum = 0.0
n = 0

if gold_cid in top_cids:
    rank = top_cids.index(gold_cid) + 1  # 1-based
    mrr_sum += 1.0 / rank
    for k in ks_sorted:
        if rank <= k:
            recall_counts[k] += 1

out = {
    "recall": {f"Recall@{k}": recall_counts[k] / n for k in ks_sorted},
    "MRR": mrr_sum / n,
}
```

### 第 3 周：重排 + 生成 + 故障处理
1. 接入重排器（可选 FlagEmbedding Cross-Encoder；没装则跳过）
2. 生成端 Prompt 约束 + 阈值拒答模板
3. 汇总至少 10 个 badcase：空召回、答非所问、幻觉、延迟高、表格解析错误等
4. 每个 badcase 都要写“根因 + 你改了什么 + 指标如何变化”

对应重要代码：为便于阅读，README 里只截取关键片段（有的地方用 `...` 表示省略），你面试讲的时候可以直接对照仓库里的完整实现文件。

- `src/dfe_rag_assistant/reranking.py`（来自 `maybe_rerank`：可选 Cross-Encoder 重排；默认按 `fusion_score` 降级）

```python
# 来自：src/dfe_rag_assistant/reranking.py -> maybe_rerank
if not cfg.paths.rerank_enabled:
    return sorted(candidates, key=lambda x: float(x.get("fusion_score", 0.0)), reverse=True)

from FlagEmbedding import FlagReranker
reranker = FlagReranker(cfg.paths.rerank_model_name, use_fp16=True)
pairs = [(query, c["content"]) for c in candidates]
scores = reranker.compute_score(pairs)

for c, s in zip(candidates, scores):
    c["rerank_score"] = float(s)
return sorted(candidates, key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
```

- `src/dfe_rag_assistant/generation.py`（来自 `build_rag_prompt` / `RagGenerator.answer`：证据约束 + 阈值拒答）

```python
# 来自：src/dfe_rag_assistant/generation.py
REFUSAL_TEXT = "根据现有资料，我无法确定这个问题的答案。建议你查阅相关技术文档或咨询专业人员。"

def build_rag_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    prompt = (
        "你是一个专业的能源装备领域技术助手。\n"
        "你只能根据“参考资料”中的内容回答问题，不得编造。\n"
        "如果参考资料不足以确定答案，请输出：\n"
        f"'{REFUSAL_TEXT}'\n\n"
        "参考资料：\n"
        f"{context}\n\n"
        f"问题：{query}\n"
        "回答："
    )
    return prompt

class RagGenerator:
    def answer(self, query: str, candidates: List[Dict[str, Any]]) -> GenerationResult:
        confidence = get_max_confidence(candidates)  # 用 fusion/rerank 得分做置信代理
        if confidence < self.cfg.generation.confidence_threshold:
            return GenerationResult(answer=REFUSAL_TEXT, confidence=confidence, sources=[])

        context_chunks = candidates[: self.cfg.retrieval.final_top_k]
        prompt = build_rag_prompt(query, context_chunks=context_chunks)
        ans = self.backend.generate(prompt)
        return GenerationResult(answer=ans, confidence=confidence, sources=[...])
```

- `src/dfe_rag_assistant/pipeline.py`（来自 `chat`：badcase 复盘时需要的“证据链字段”来源于这里）

```python
# 来自：src/dfe_rag_assistant/pipeline.py -> chat
return {
    "sources": result.sources,  # chunk_id / pdf_path / page_num / element_type
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
```

### 第 4 周：评测闭环 + Demo + 文档化
1. RAGAS/自建离线评测（或只做检索 Recall + 生成人工抽检）
2. 搭建 Gradio/Streamlit Demo：展示检索 chunk、引用来源、拒答效果
3. 整理项目文档：架构图、选型理由、实验表、badcase 池

对应重要代码：为便于阅读，README 里只截取关键片段（有的地方用 `...` 表示省略），你面试讲的时候可以直接对照仓库里的完整实现文件。

- `src/dfe_rag_assistant/eval_retrieval.py`：离线评测闭环的指标计算入口（`Recall@K / MRR / NDCG`）
- `demo_gradio.py`：Demo 前端入口（调用 `chat()` 并返回 `answer/sources`）
- `src/dfe_rag_assistant/indexing.py`：文档化所需“可复现依据”
  - `run_config.json`：保存 chunk/embedding 等配置快照
  - `chunks_meta.json`：保存 chunk 的内容与来源，便于你写引用溯源与失败样本分析

```python
# 来自：src/dfe_rag_assistant/indexing.py（落盘文件）
_save_json(index_path / cfg.paths.chunks_meta_json, all_chunks)
_save_json(index_path / "run_config.json", asdict(cfg))
```

## 3. 你在面试时应该重点叙述什么（建议按“做了什么 + 为什么有效 + 怎么验证”）

建议你把你的贡献点集中在 4 个模块上，并都要带数字或现象：
1. **文档处理与切块**：表格保护 + 规则边界，避免参数表被切断；用 chunk 实验找到最优组合（如 512/100）。
2. **混合召回**：Dense + BM25 + RRF 融合；用评测集按类型拆解，证明对“对比/汇总”更稳。
3. **重排与证据约束**：重排把候选证据质量抬上去；生成端加阈值拒答和引用约束降低幻觉。
4. **评测闭环**：自建评测集 + badcase 池持续迭代，而不是只做一次 Demo。

## 4. 代码仓库怎么运行（Demo/构建/评测）

在仓库根目录执行（Windows PowerShell）：

```powershell
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### 4.1 构建索引
需要：
- `--pdf_dir`：存放 PDF 的目录
- `--index_dir`：索引输出目录（会生成 FAISS/BM25 索引和 metadata）

示例：

```powershell
python -m src.dfe_rag_assistant build-index --pdf_dir "data/pdfs" --index_dir "data/index_dfe"
```

### 4.2 交互问答
```powershell
python -m src.dfe_rag_assistant chat --index_dir "data/index_dfe" --query "风机叶片多长？"
```

### 4.3 评测检索效果（Recall@K）
准备 `evalset.jsonl`（每行 JSON），至少包含：
- `query`
- `gold_chunk_id`（必须与索引里的 chunk_id 对应）

```powershell
python -m src.dfe_rag_assistant eval --index_dir "data/index_dfe" --eval_jsonl "data/evalset.jsonl"
```

## 5. 后续你可以怎么扩展（多模态岗也能用得上）

虽然这个项目主要是“文本 RAG”，但你可以在面试时把它迁移到多模态能力上：
- 多模态切块：把图表/公式/表格区域做 OCR 或结构化抽取，作为“视觉证据 chunk”
- 多模态检索：文本 embedding + 图像 embedding 联合召回，再做 rerank
- 多跳推理：复杂问题先找图表/结构化证据，再由模型汇总成答案

你在回答“为什么你适合多模态算法岗”时，核心逻辑是：你已经把“文档解析->结构化->召回->重排->评测闭环”做成了工程化流程，这和多模态 pipeline 的思路高度一致。

