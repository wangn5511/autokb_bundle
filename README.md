# Auto-KB

一键在 Linux 服务器上跑通的 RAG Demo：PDF/MD/TXT 摄取、BM25+向量融合、Cross-Encoder 重排、可溯源引用和置信度/拒答、FastAPI + Gradio 前端、支持增量索引。

## 快速开始
```bash
unzip autokb_bundle.zip && cd autokb
bash scripts/run.sh
# 浏览器访问:
#   http://<你的公网IP>:8000/docs
#   http://<你的公网IP>:7860
```

## 目录
```
app/            # 业务代码（ingest / retriever / llm / server / ui）
scripts/run.sh  # 一键启动脚本
data/your_docs/ # 放你的 PDF/MD/TXT
indexes/        # 索引产物
logs/           # 日志
reports/        # 评测输出
requirements.txt
.env.example
```
