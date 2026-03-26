from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from .config import AppConfig


REFUSAL_TEXT = "根据现有资料，我无法确定这个问题的答案。建议你查阅相关技术文档或咨询专业人员。"


def build_rag_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Prompt 约束：只基于参考资料回答；无法确定则拒答。
    """
    context_lines: List[str] = []
    for i, c in enumerate(context_chunks, start=1):
        pdf_name = c.get("pdf_path", "")
        page_num = c.get("page_num", "")
        context_lines.append(f"[文档{ i } | {pdf_name} | 第{page_num}页]\n{c['content']}")

    context = "\n\n".join(context_lines)
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


@dataclass
class GenerationResult:
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]


class TransformersGenerator:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.generation.llm_model_name, trust_remote_code=True)
        # 自动设备映射：GPU 可用则上 GPU，否则可能慢（取决于你的环境）
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.generation.llm_model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    def generate(self, prompt: str) -> str:
        from transformers import GenerationConfig as HFGenerationConfig

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_cfg = HFGenerationConfig(
            max_new_tokens=self.cfg.generation.max_new_tokens,
            do_sample=True,
            temperature=self.cfg.generation.temperature,
            top_p=self.cfg.generation.top_p,
            repetition_penalty=self.cfg.generation.repetition_penalty,
        )
        out = self.model.generate(**inputs, generation_config=gen_cfg)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # 简化：直接返回“回答：”后面的尾部（用于面试 Demo 足够）
        return text.split("回答：")[-1].strip()


class VLLMGenerator:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        from vllm import LLM, SamplingParams

        self.sampling_params = SamplingParams(
            temperature=cfg.generation.temperature,
            top_p=cfg.generation.top_p,
            max_tokens=cfg.generation.max_new_tokens,
            repetition_penalty=cfg.generation.repetition_penalty,
        )
        self.llm = LLM(
            model=cfg.generation.llm_model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            trust_remote_code=True,
        )

    def generate(self, prompt: str) -> str:
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text.strip()


def get_max_confidence(candidates: List[Dict[str, Any]]) -> float:
    scores = []
    for c in candidates:
        if "rerank_score" in c and c["rerank_score"] is not None:
            scores.append(float(c["rerank_score"]))
        elif "fusion_score" in c:
            scores.append(float(c["fusion_score"]))
    return max(scores) if scores else 0.0


class RagGenerator:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        if cfg.generation.llm_backend == "vllm":
            try:
                self.backend = VLLMGenerator(cfg)
            except Exception:
                # vLLM 可能在 Windows 环境不可用：回退到 transformers
                self.backend = TransformersGenerator(cfg)
        else:
            self.backend = TransformersGenerator(cfg)

    def answer(self, query: str, candidates: List[Dict[str, Any]]) -> GenerationResult:
        # 拒答：根据融合/重排得分上界做置信代理
        confidence = get_max_confidence(candidates)
        if confidence < self.cfg.generation.confidence_threshold:
            return GenerationResult(answer=REFUSAL_TEXT, confidence=confidence, sources=[])

        # 取重排后 Top-N 候选作为上下文（默认就是 final_top_k=5）
        context_chunks = candidates[: self.cfg.retrieval.final_top_k]
        prompt = build_rag_prompt(query, context_chunks=context_chunks)
        ans = self.backend.generate(prompt)

        sources: List[Dict[str, Any]] = []
        for c in context_chunks:
            sources.append(
                {
                    "chunk_id": c.get("chunk_id"),
                    "pdf_path": c.get("pdf_path"),
                    "page_num": c.get("page_num"),
                    "element_type": c.get("element_type"),
                }
            )
        return GenerationResult(answer=ans, confidence=confidence, sources=sources)

