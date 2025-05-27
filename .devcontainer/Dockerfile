FROM python:3.10-slim

# 基本パッケージのインストール
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ設定
WORKDIR /workspace

# 必要なPythonパッケージをインストール
RUN pip install --upgrade pip && \
    pip install notion-client langchain langchain-text-splitters \
    langchain-community faiss-cpu sentence-transformers \
    fastapi uvicorn gradio ollama

# 環境変数設定
ENV PYTHONPATH=/workspace