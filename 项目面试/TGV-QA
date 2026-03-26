# 📘 多模态 / LLM 岗面试问答大全

> 适用于：多模态大模型 / LLM工程师 / GraphRAG / AI算法岗  
> 内容涵盖：LLM基础、RAG/GraphRAG、多模态、工程部署、项目实战

---

## 🧠 一、LLM基础

### ❓1 LLM 的基本原理是什么？

**回答：**
- 基于 Transformer 的自回归语言模型
- 通过预测下一个 token 学习语言分布
- 利用 self-attention 捕捉上下文依赖

**扩展：**
- Scaling Law
- 预训练 + SFT + RLHF

---

### ❓2 Transformer 为什么有效？

**回答：**
- self-attention 可建模任意位置依赖
- 并行计算能力强
- 表达能力优于 RNN

---

### ❓3 什么是 hallucination？如何解决？

**回答：**
- 模型生成不真实但合理的内容

**解决方法：**
- RAG（引入外部知识）
- GraphRAG（结构约束）
- Prompt约束
- 后处理校验

---

## 🧠 二、RAG / GraphRAG

### ❓4 什么是 RAG？

**回答：**
- 检索增强生成（Retrieval-Augmented Generation）
- 检索知识 + LLM生成

---

### ❓5 GraphRAG 和 RAG 区别？

| 传统RAG | GraphRAG |
|--------|----------|
| 文本检索 | 图结构检索 |
| 无逻辑 | 因果关系 |
| 不可解释 | 可解释 |

**总结：**
GraphRAG 适用于因果推理场景（如工业）

---

### ❓6 GraphRAG 实现流程

```text
用户问题
  ↓
实体识别
  ↓
图谱检索
  ↓
路径搜索（多跳）
  ↓
构造Prompt
  ↓
LLM推理

❓7 为什么 GraphRAG 更适合工业？

工业问题是“因果问题”

GraphRAG支持溯因推理

提供可解释路径

🧠 三、多模态
❓8 什么是多模态模型？

同时处理图像 / 文本 / 语音

共享语义空间

❓9 CLIP 原理

图像 & 文本编码到同一空间

使用对比学习（contrastive learning）

使用 cosine similarity

❓10 多模态对齐方法

对比学习（CLIP）

共享 embedding 空间

cross-attention

🧠 四、工程实现
❓11 如何部署大模型？
ollama pull qwen3.5:27b

使用 Ollama 本地部署

支持离线推理（工业安全）

❓12 如何提高推理效率？

模型量化（int8 / int4）

KV Cache

Prompt压缩

Batch推理

❓13 Prompt设计原则

明确任务（instruction）

提供结构化上下文（Graph路径）

限制输出格式

🧠 五、项目相关（GraphRAG）
❓14 项目创新点

因果知识图谱（非普通KG）

GraphRAG 推理

LLM + 工业场景结合

❓15 解决了什么问题？

工业缺陷分析依赖人工经验

提供自动化 + 可解释推理

❓16 如何优化系统？

引入多模态数据（图像）

加权知识图谱

强化学习优化路径

🧠 六、进阶问题
❓17 为什么不用纯 LLM？

LLM 易 hallucination

GraphRAG 提供结构约束

❓18 RAG 瓶颈

检索质量

上下文长度

信息冗余

❓19 多模态未来方向

统一模型（文本+视觉+视频）

强推理能力

工业落地

🎯 总结（面试必背）
我主要研究多模态与大模型在工业场景的应用，
基于因果知识图谱与GraphRAG，
结合Qwen3.5实现可解释推理，
解决传统方法依赖经验的问题。
