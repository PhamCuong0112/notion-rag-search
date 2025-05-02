import gradio as gr
import logging
import sys
import traceback
import re
from typing import List, Tuple

from app.rag.orchestrator import RAGOrchestrator
from app.llm.ollama import OllamaClient

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_gradio_app():
    """Gradioチャットアプリを作成"""
    
    # RAGとLLMコンポーネントの初期化
    rag = RAGOrchestrator()
    llm = OllamaClient()
    
    def clean_response(text: str) -> str:
        """LLMの応答から<think>タグとその内容を除去する これをやらないとchat画面には何も出ない"""
        # <think>...</think>パターンを削除
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # タグが閉じられていない場合（<think>...のみの場合）も削除
        cleaned_text = re.sub(r'<think>.*', '', cleaned_text, flags=re.DOTALL)
        
        # 先頭と末尾の空白を削除
        cleaned_text = cleaned_text.strip()
        
        # 空の応答の場合はデフォルトメッセージを返す
        if not cleaned_text:
            return "申し訳ありませんが、適切な回答を生成できませんでした。"
            
        return cleaned_text
    
    def respond(message: str, history: List[Tuple[str, str]]):
        """チャットボットの応答関数"""
        try:
            logger.info(f"ユーザーメッセージを受信: {message}")
            
            # 関連コンテキストを取得
            logger.info("RAGからコンテキストを取得中...")
            contexts, sources = rag.retrieve(message)
            logger.info(f"取得したコンテキスト数: {len(contexts)}")
            logger.info(f"取得したソース数: {len(sources)}")
            
            # 履歴をLLM用に変換
            formatted_history = []
            for user_msg, assistant_msg in history:
                formatted_history.append({"role": "user", "content": user_msg})
                formatted_history.append({"role": "assistant", "content": assistant_msg})
            
            # 回答を生成
            logger.info("LLMから回答を生成中...")
            response = llm.generate_response(message, contexts, history=formatted_history)
            logger.info(f"LLMから回答を受信: {response[:1000]}...")  # 回答の先頭部分をログに出力
            
            # 応答をクリーニング
            cleaned_response = clean_response(response)
            logger.info(f"クリーニング後の応答: {cleaned_response[:1000]}...")
            
            # ソース情報がある場合は追加
            if sources:
                cleaned_response += "\n\n**参考情報:**\n"
                for i, source in enumerate(sources, 1):
                    cleaned_response += f"{i}. {source}\n"
            
            logger.info("応答を返します")
            return cleaned_response
        except Exception as e:
            # 例外情報を詳細にログに出力
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            logger.error(f"回答生成中にエラーが発生: {str(e)}\n{tb_str}")
            
            # ユーザーにもエラー情報を返す
            return f"回答生成中にエラーが発生しました: {str(e)}"
    
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
    )
    
    return chat_interface

# Gradioアプリの起動コード
if __name__ == "__main__":
    app = create_gradio_app()
    logger.info("Gradioアプリを起動します...")
    app.launch(server_name="0.0.0.0", share=False, debug=True)  # デバッグモードを有効化