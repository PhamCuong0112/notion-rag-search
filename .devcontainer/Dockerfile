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

# Ollamaのインストール
RUN curl -fsSL https://ollama.com/install.sh | sh

# 起動スクリプトの作成（コンテナ起動時にOllamaを開始し、必要なモデルをダウンロード）
RUN echo '#!/bin/bash\n\
echo "Starting Ollama server..."\n\
ollama serve > /var/log/ollama.log 2>&1 &\n\
sleep 5\n\
echo "Checking if llama2 model is available..."\n\
if ! ollama list | grep -q "llama2"; then\n\
  echo "Downloading llama2 model..."\n\
  ollama pull llama2\n\
fi\n\
echo "Ollama setup completed."\n\
exec "$@"' > /usr/local/bin/startup.sh && \
    chmod +x /usr/local/bin/startup.sh

# 環境変数設定
ENV PYTHONPATH=/workspace

# コンテナ起動時にスタートアップスクリプトを実行
ENTRYPOINT ["/usr/local/bin/startup.sh"]
CMD ["bash"]