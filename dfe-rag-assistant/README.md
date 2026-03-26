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

### 第 2 周：Query 理解 + 多路召回 + 召回评测
1. 实现轻量 Query 扩展（同义词规则 + BM25 分词天然覆盖）
2. 实现 3 路召回（向量/BM25/融合 RRF）
3. 自建评测集（query -> gold chunk/doc）
4. 产出评测表：Recall@3/5/10，按问题类型（事实/对比/汇总）分桶分析

### 第 3 周：重排 + 生成 + 故障处理
1. 接入重排器（可选 FlagEmbedding Cross-Encoder；没装则跳过）
2. 生成端 Prompt 约束 + 阈值拒答模板
3. 汇总至少 10 个 badcase：空召回、答非所问、幻觉、延迟高、表格解析错误等
4. 每个 badcase 都要写“根因 + 你改了什么 + 指标如何变化”

### 第 4 周：评测闭环 + Demo + 文档化
1. RAGAS/自建离线评测（或只做检索 Recall + 生成人工抽检）
2. 搭建 Gradio/Streamlit Demo：展示检索 chunk、引用来源、拒答效果
3. 整理项目文档：架构图、选型理由、实验表、badcase 池

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

