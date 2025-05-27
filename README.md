# Notion RAG検索システム

Notionのマニュアルやドキュメントをベクトルデータベースに保存し、自然言語の質問に対して関連情報を検索・回答するシステムです。RAG（Retrieval-Augmented Generation）技術を使用して、LLM（大規模言語モデル）の回答をNotionの情報をもとに回答します。

## 動かした時のイメージ
![デモアニメーション](.github/images/chatbot.gif)

## Deepwiki by Devin
https://deepwiki.com/yukiyoshimura/notion-rag-search

## 技術スタック

### バックエンド
- **FastAPI**: APIエンドポイントの提供
- **Uvicorn**: ASGIサーバー(Asynchronous Server Gateway Interface)

### Notion連携
- **Notion API**: Notionのページ内容を取得するためのクライアント

### RAG（Retrieval-Augmented Generation）
- **FAISS**: 高速なベクトル類似度検索エンジン
- **HuggingFace埋め込みモデル**: multilingual-e5-small
- **LangChain**: テキスト分割・処理のためのツール

### LLM（大規模言語モデル）
- **Ollama**: ローカルLLMの実行エンジン
- **Qwen3-4B**: 軽量で多言語対応のLLMモデル

### UI
- **Gradio**: チャットインターフェースの提供

## セットアップ

### 環境変数の設定

`.env`ファイルを作成し、以下の環境変数を設定します。

```
# Notion API設定
NOTION_TOKEN=your_notion_integration_token
NOTION_PAGE_ID=your_notion_parent_page_id

# LLM設定（必要に応じて変更）
LLM_MODEL=qwen3:4b
OLLAMA_API_BASE=http://host.docker.internal:11434/api

# 埋め込みモデル（必要に応じて変更）
EMBEDDING_MODEL=intfloat/multilingual-e5-small

# RAG設定（必要に応じて調整）
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
```

### 必要パッケージのインストール

```bash
pip install -r requirements.txt
```

### Ollamaのインストールと実行

このシステムはOllamaを使用してローカルでLLMを実行します。以下の手順でOllamaをセットアップしてください。

1. **Ollamaのインストール**:
   - [Ollama公式サイト](https://ollama.com/download)から、お使いのOSに合わせたインストーラをダウンロードしてインストールします。
   - Linux、macOS、Windowsに対応しています。

2. **必要なモデルのダウンロード**:
   ```bash
   # デフォルトで使用するqwen3:4bモデルをダウンロード
   ollama pull qwen3:4b
   ```
   - 初回実行時は数GB程度のモデルファイルがダウンロードされます。
   - 別のモデルを使用する場合は、モデル名を変更して実行してください。

3. **Ollamaサーバーの起動**:
   ```bash
   # Ollamaサーバーを起動（バックグラウンドで実行）
   ollama serve
   ```
   - デフォルトでは http://localhost:11434 でサービスが提供されます。
   - 環境変数 `OLLAMA_HOST` を設定することで、ホストやポートを変更できます。

Dockerコンテナ内からOllamaにアクセスする場合、`.env`ファイルの`OLLAMA_API_BASE`が`http://host.docker.internal:11434/api`に設定されていることを確認してください。

## 使用方法

### 1. Notionデータのベクトル化とインデックス構築

Notionのページ内容を取得し、ベクトルデータベースにインデックス化します。

```bash
python scripts/build_index.py
```

* 既存のインデックスを上書きする場合は `--force` オプションを追加してください。
* 成功すると `data/index.faiss` と `data/documents.pkl` ファイルが生成されます。

### 2. バッチモードでの動作確認

コマンドラインから特定のクエリに対する応答をテストします。

```bash
python scripts/test_query.py "検索したいクエリ"
```

* 関連コンテキストも表示したい場合は `--show-contexts` オプションを追加してください。

### 3. チャットボットインターフェースの起動

Gradioベースのチャットインターフェースを起動します。

```bash
python -m app.ui.gradio_app
```

ブラウザで http://localhost:7860 にアクセスして、チャットインターフェースを使用できます。

## プロジェクト構造

```
notion-rag-search/
├── app/                    # アプリケーションコード
│   ├── __init__.py
│   ├── main.py             # アプリケーションのエントリーポイント
│   ├── api/                # API定義
│   │   ├── __init__.py
│   │   └── endpoints.py    # FastAPIのエンドポイント
│   ├── core/               # コア機能
│   │   ├── __init__.py
│   │   ├── config.py       # 設定管理
│   │   └── notion.py       # Notion APIクライアント
│   ├── rag/                # RAG実装
│   │   ├── __init__.py
│   │   ├── embedding.py    # テキスト埋め込み処理
│   │   ├── vector_store.py # FAISSベクトルストア
│   │   └── orchestrator.py # RAG検索オーケストレーター
│   ├── llm/                # LLM関連
│   │   ├── __init__.py
│   │   └── ollama.py       # Ollamaクライアント
│   └── ui/                 # ユーザーインターフェース
│       ├── __init__.py
│       └── gradio_app.py   # Gradioチャットインターフェース
├── scripts/                # ユーティリティスクリプト
│   ├── build_index.py      # インデックス構築スクリプト
│   └── test_query.py       # クエリテストスクリプト
├── data/                   # 生成されるデータファイル
│   ├── index.faiss         # FAISSインデックスファイル
│   └── documents.pkl       # ドキュメントデータ
├── requirements.txt        # 依存パッケージ
└── README.md               # このファイル
```

## 機能と特徴

- **Notion連携**: 指定した親ページとその子ページを再帰的に取得
- **チャンク分割**: ドキュメントを最適なサイズに分割して検索精度を向上
- **ベクトル検索**: 高速なFAISSによる類似度検索
- **ソース引用**: 回答の根拠となった情報源を表示

## 注意事項

- Ollamaはホストマシンで実行する必要があります（デフォルト設定は `host.docker.internal` を参照）
- 大量のデータをインデックス化する場合はメモリ使用量に注意してください
- LLMモデルの選択によって回答の品質と速度が変わります