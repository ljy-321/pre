一、视觉-语言遥感变化检测（CLIP + Change Detection）
1. 关于 CLIP

Q1：CLIP 是什么？为什么适合用在遥感变化检测？
答：
CLIP 是一个视觉-语言对齐模型，通过对比学习将图像和文本映射到同一语义空间。

在遥感变化检测中，它的优势是：

提供高层语义先验（如“建筑物变化”“道路新增”）
弥补 CNN 只关注像素差异的问题
能区分：
真实变化（建筑新增）
伪变化（光照、阴影）

👉 本质：从“像素差异”提升到“语义差异”

Q2：CLIP 的 text embedding 在你项目中起什么作用？
答：
我把 text embedding 作为语义锚点（semantic prior），用于引导视觉特征：

将图像特征投影到 CLIP 空间
与文本 embedding 做对齐（alignment loss）
强化“变化类别”的语义表达

👉 相当于：

用语言告诉模型“什么叫变化”

2. Cross-Attention

Q3：为什么用 cross-attention 融合双时相特征？
答：
传统方法（concat / 差分）的问题：

无法建模跨时相依赖关系
容易引入噪声

cross-attention 优势：

A 时相作为 Query，B 时相作为 Key/Value
自动关注变化区域
抑制无关区域

👉 本质：

“让一个时间点去主动查询另一个时间点的变化”

3. 软差异图（Soft Difference Map）

Q4：软差异图和直接做 feature difference 有什么区别？
答：

方法	问题
直接差分	对噪声敏感
软差异图	可学习权重

我的方法：

用 attention / similarity 生成差异权重
输出连续值（不是硬差）

👉 优点：

更鲁棒
可微、可学习
4. 空间中心先验（Spatial Prior）

Q5：什么是空间中心先验？为什么有效？
答：
在遥感变化中：

变化通常呈局部聚集（cluster）

我引入：

中心区域权重增强
提高变化区域响应

👉 类似：

给模型一个“变化通常成块出现”的归纳偏置

5. 小目标检测提升

Q6：为什么你的方法能提升小目标检测？
答：

主要原因三点：

CLIP 提供语义增强（弱小目标更容易被识别）
抑制未变化区域（减少 false positive）
attention 聚焦关键区域

👉 本质：

提高信噪比（Signal / Noise）

二、RAG 企业问答系统
6. RAG

Q7：RAG 的核心思想是什么？
答：
RAG = 检索 + 生成

流程：

用户问题
检索相关文档
拼接 prompt
LLM 生成答案

👉 本质：

用外部知识增强 LLM

7. BM25 + Dense Retrieval

Q8：为什么要用 BM25 + Dense 混合召回？
答：

方法	优点	缺点
BM25	精准关键词	无语义
Dense	语义强	容易偏

融合后：

兼顾 lexical + semantic

👉 Recall 提升的关键点

8. Cross-Encoder 重排

Q9：Cross-Encoder 和 Bi-Encoder 有什么区别？
答：

模型	特点
Bi-Encoder	向量检索快
Cross-Encoder	精排准确

Cross-Encoder：

输入 query + doc 一起编码
输出相关性分数

👉 用于 rerank top-k

9. 表格保护切块策略

Q10：为什么要做“表格保护”？
答：
PDF 中：

表格容易被拆碎
语义破坏严重

我的策略：

表格整体切块
避免跨 chunk

👉 提升：

Recall
可读性
10. Qwen2.5-7B + vLLM

Q11：为什么用 vLLM？
答：
vLLM 的优势：

PagedAttention（高效 KV cache）
支持高并发
显存利用率高

👉 结果：

QPS 提升
latency 降低

Q12：如何降低幻觉？
答：

我用了三种方法：

检索约束（必须引用）
同义词扩展（减少漏召回）
拒答机制（no evidence → no answer）

👉 工程上很关键

三、GraphRAG + 因果推理
11. GraphRAG

Q13：GraphRAG 和传统 RAG 的区别？
答：

RAG	GraphRAG
文本块	图结构
无关系	有因果关系
单跳检索	多跳推理

👉 GraphRAG 更适合：

工业知识
因果分析
12. 因果知识图谱

Q14：为什么要用因果图谱而不是普通知识图谱？
答：

普通 KG：

只是关系（A related to B）

因果 KG：

A → B（因果）

👉 优势：

可解释
可推理
可溯因（root cause）
13. 多跳推理

Q15：什么是多跳推理？
答：

例如：

工艺参数 → 温度变化 → 材料应力 → 裂纹缺陷

👉 需要多步 reasoning，而不是一次检索

14. Qwen3-27B + Ollama

Q16：为什么用本地部署（Ollama）？
答：

数据隐私（工业数据）
可控性强
成本低

👉 企业场景必问点

15. 因果推理流程

Q17：你的推理流程有什么创新点？
答：

流程：

结构化检索
因果路径召回
LLM 约束生成
解释输出

👉 关键创新：

LLM 被约束，而不是自由生成
四、综合类高频问题（必问）
Q18：你三个项目的共同核心能力是什么？

答：

我总结为三点：

多模态融合能力
CLIP（视觉+语言）
RAG（文本+知识）
结构化增强 LLM
GraphRAG
因果图谱
工程落地能力
vLLM 高并发
实际 QPS 优化
Q19：你最大的技术亮点是什么？

答：

不是单点创新，而是：

👉 “结构化信息 + 大模型”结合

CLIP → 语义结构
RAG → 知识结构
Graph → 因果结构
Q20：如果让你再优化一个点，你会改哪里？

答：

可以答：

Change Detection → 引入 SAM / Segment Anything
RAG → 引入 Agent（多步推理）
GraphRAG → 自动构图（减少人工）
最后给你一个建议（很关键）

你这三段项目已经很强了，但面试官真正想听的是：

👉 你是否理解“为什么这么设计”

而不是：

用了什么模型
做了哪些模块
