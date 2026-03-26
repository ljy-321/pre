from __future__ import annotations

import json

import gradio as gr

from src.dfe_rag_assistant.config import AppConfig
from src.dfe_rag_assistant.pipeline import chat


def _run(query: str, index_dir: str, llm_backend: str, llm_model: str):
    cfg = AppConfig()
    cfg.generation.llm_backend = llm_backend
    cfg.generation.llm_model_name = llm_model
    # demo 默认走 5 个证据片段
    out = chat(query=query, index_dir=index_dir, cfg=cfg)
    return out["answer"], out["sources"]


with gr.Blocks(title="东方电气智能培训助手（RAG Demo）") as demo:
    gr.Markdown("## 输入问题，系统将展示引用证据并生成答案")

    with gr.Row():
        index_dir = gr.Textbox(label="index_dir", value="data/index_dfe")
        llm_backend = gr.Dropdown(label="llm_backend", choices=["transformers", "vllm"], value="transformers")
        llm_model = gr.Textbox(label="llm_model", value="Qwen/Qwen2.5-7B-Instruct")

    query = gr.Textbox(label="Query", placeholder="例如：风机叶片材料是什么？", lines=2)
    btn = gr.Button("生成回答")

    answer = gr.Textbox(label="Answer", lines=6)
    sources = gr.JSON(label="Sources（引用证据）")

    btn.click(fn=_run, inputs=[query, index_dir, llm_backend, llm_model], outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

