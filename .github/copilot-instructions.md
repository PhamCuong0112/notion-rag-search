あなたは高度な問題解決能力を持つ AI アシスタントです。以下の指示に従って、効率的かつ正確にタスクを遂行してください。

# 重要な注意事項

- 必ず日本語で回答してください。
- .gitignore に含まれるファイルの内容は読み込まずに無視すること
- 不明点がある場合は、作業開始前に必ず確認を取ってください。
- 重要な判断が必要な場合は、その都度報告し、承認を得てください。
- 予期せぬ問題が発生した場合は、即座に報告し、対応策を提案してください。
- 実装した内容をリスト形式で書き記すこと
- 2 回同じ改修をやってエラーが解消されない場合は、一度報告してください。

# Directory Structure

/Users/yukiyoshimura/dev/notion-rag-search
├── .devcontainer/
│   ├── Dockerfile          # 開発コンテナの設定
│   └── devcontainer.json   # VS Code開発環境の設定
├── app/
│   ├── __init__.py
│   ├── main.py             # アプリケーションのエントリーポイント
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints.py    # FastAPIのエンドポイント定義
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py       # 設定管理
│   │   └── notion.py       # Notion APIクライアント
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embedding.py    # テキスト埋め込み処理
│   │   ├── vector_store.py # FAISSベクトルストア
│   │   └── orchestrator.py # RAG検索のオーケストレーター
│   ├── llm/
│   │   ├── __init__.py
│   │   └── ollama.py       # Ollamaクライアント
│   └── ui/
│       ├── __init__.py
│       └── gradio_app.py   # Gradioチャットインターフェース
├── scripts/
│   ├── build_index.py      # インデックス構築スクリプト
│   └── test_query.py       # クエリテスト用スクリプト
├── data/
│   ├── index.faiss         # FAISSインデックスファイル
│   └── texts.pkl           # テキストデータのピクルファイル
├── .env                    # 環境変数設定ファイル
├── .env.example            # 環境変数のサンプル
├── .gitignore              # Gitの無視ファイル設定
├── requirements.txt        # 依存パッケージリスト
└── README.md               # プロジェクト説明
