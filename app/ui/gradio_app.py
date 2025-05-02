import gradio as gr
from typing import List, Tuple

from app.rag.orchestrator import RAGOrchestrator
from app.llm.ollama import OllamaClient

def create_gradio_app():
    """Gradioチャットアプリを作成"""
    
    # RAGとLLMコンポーネントの初期化
    rag = RAGOrchestrator()
    llm = OllamaClient()
    
    def respond(message: str, history: List[Tuple[str, str]]):
        """チャットボットの応答関数"""
        # 関連コンテキストを取得
        contexts, sources = rag.retrieve(message)
        
        # 履歴をLLM用に変換
        formatted_history = []
        for user_msg, assistant_msg in history:
            formatted_history.append({"role": "user", "content": user_msg})
            formatted_history.append({"role": "assistant", "content": assistant_msg})
        
        # 回答を生成
        response = llm.generate_response(message, contexts, history=formatted_history)
        
        # ソース情報がある場合は追加
        if sources:
            response += "\n\n**参考情報:**\n"
            for i, source in enumerate(sources, 1):
                response += f"{i}. {source}\n"
        
        return response
    
    # Gradioインターフェースの作成
    chat_interface = gr.ChatInterface(
        respond,
        title="Notionマニュアル検索チャットボット",
        description="マニュアルに基づいて質問に回答します。",
        theme="soft",
        examples=[
            "マニュアルの使い方を教えてください",
            "設定方法はどうすればいいですか？",
            "エラーが発生した場合の対処法は？"
        ],
        retry_btn=None,
        undo_btn=None,
        clear_btn="会話をクリア",
    )
    
    return chat_interface