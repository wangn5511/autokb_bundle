# -*- coding: utf-8 -*-
import os, requests
from typing import List, Dict, Any

def simple_query_rewrite(q: str) -> List[str]:
    qs = {q}
    if "怎么" in q or "如何" in q:
        qs.add(q.replace("怎么", "如何"))
    if "安装" in q:
        qs.add(q + " 步骤")
        qs.add(q + " 依赖")
    if len(q) < 15:
        qs.add(q + " 详细说明")
    return list(qs)

def llm_query_rewrite(q: str) -> List[str]:
    if os.getenv("OPENAI_API_KEY"):
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
        prompt = f"请基于检索改写以下中文问题，输出3个不同但等价的搜索query：{q}"
        data = {"model": model, "input": prompt}
        try:
            r = requests.post(f"{base}/responses", json=data, headers=headers, timeout=20)
            r.raise_for_status()
            txt = r.json().get("output_text", "")
            lines = [l.strip("- ").strip() for l in txt.split("\n") if l.strip()]
            return lines[:3] if lines else simple_query_rewrite(q)
        except:
            return simple_query_rewrite(q)
    return simple_query_rewrite(q)

def generate_answer(contexts: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
    citations = [{
        "source": c["meta"].get("source"),
        "page": c["meta"].get("page"),
        "snippet": c["text"][:220]
    } for c in contexts]

    conf = sum([c["score"] for c in contexts[:3]]) / max(1, min(3, len(contexts)))

    if os.getenv("OPENAI_API_KEY"):
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
        sys = "你是严谨的企业知识助理。只根据提供的上下文回答，若不确定请说不确定。用中文回答，并引用事实。"
        ctx_text = "\n\n".join([f"[{i}] {c['text']}" for i,c in enumerate(contexts)])
        user = f"问题：{question}\n仅使用以上上下文作答，并尽量引用 [索引号]。"
        data = {"model": model, "input": [{"role":"system","content":sys},{"role":"user","content":f"{ctx_text}\n\n{user}"}]}
        try:
            r = requests.post(f"{base}/responses", json=data, headers=headers, timeout=30)
            r.raise_for_status()
            ans = r.json().get("output_text","").strip()
            return {"answer": ans, "citations": citations, "confidence": float(conf)}
        except:
            pass

    top = contexts[0]["text"] if contexts else ""
    return {"answer": top[:400], "citations": citations, "confidence": float(conf)}
