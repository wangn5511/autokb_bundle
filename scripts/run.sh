#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) venv & deps
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt

# 2) 环境
if [ ! -f ".env" ]; then
  cp .env.example .env
fi
set -a
source .env
set +a

# 3) 首次构建（如无索引）
if [ ! -f "indexes/docstore.pkl" ]; then
  mkdir -p data/your_docs indexes logs
  echo "tips: 将你的 PDF/MD/TXT 放到 data/your_docs/"
  python -c "from app.ingest import ingest; ingest('data/your_docs','indexes',800,150,False)"
fi

# 4) 后端 & 前端
nohup uvicorn app.server:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
sleep 2
python app/ui.py
