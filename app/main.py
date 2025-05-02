# アプリケーションのエントリーポイント
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

from app.core.config import Settings
from app.api.endpoints import router
from app.ui.gradio_app import create_gradio_app

def create_app():
    # 設定の読み込み
    settings = Settings()
    
    # FastAPIアプリケーションの作成
    app = FastAPI(
        title="Notion RAG Chatbot",
        description="Notionのマニュアルを検索して質問に回答するチャットボット",
        version="1.0.0"
    )
    
    # CORSミドルウェアの追加
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # APIルーターの登録
    app.include_router(router, prefix="/api")
    
    # Gradioアプリの作成とマウント
    gradio_app = create_gradio_app()
    app = gr.mount_gradio_app(app, gradio_app, path="/")
    
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=True)