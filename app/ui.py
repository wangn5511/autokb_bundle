# -*- coding: utf-8 -*-
import os, requests, gradio as gr
API = os.getenv("AUTOKB_API", "http://127.0.0.1:8000")

def ask(q, top_k, rerank, rewrite):
    payload = {"query": q, "top_k": int(top_k), "use_rerank": bool(rerank), "rewrite": rewrite}
    r = requests.post(f"{API}/query", json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    ans = data.get("answer","")
    conf = data.get("confidence",0.0)
    cites = data.get("citations",[])
    html = f"<h3>答案（置信度 {conf:.2f}）</h3><div>{ans}</div><hr><h4>引用</h4>"
    for c in cites:
        src = c.get("source")
        page = c.get("page")
        snip = c.get("snippet","")[:300]
        badge = f"{src} / 页 {page}" if page else f"{src}"
        html += f"<div><b>{badge}</b><br><code>{snip}</code></div><hr>"
    return html

with gr.Blocks(title="Auto-KB") as demo:
    gr.Markdown("# Auto-KB 知识助理")
    with gr.Row():
        q = gr.Textbox(label="请输入问题", lines=2, value="项目的安装步骤是什么？")
    with gr.Row():
        topk = gr.Slider(3, 12, value=8, step=1, label="Top-K")
        rerank = gr.Checkbox(value=True, label="启用重排序（Cross-Encoder）")
        rewrite = gr.Radio(["off","rules","llm"], value="rules", label="Query Rewriting")
    out = gr.HTML()
    btn = gr.Button("提问")
    btn.click(ask, inputs=[q, topk, rerank, rewrite], outputs=[out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
