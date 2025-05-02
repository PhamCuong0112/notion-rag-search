import requests
import json
import logging
from typing import List, Optional, Dict, Any

from app.core.config import get_settings

class OllamaClient:
    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self.model = model_name or settings.llm_model
        self.api_base = "http://localhost:11434/api"
    
    def generate_response(self, query: str, contexts: List[str], history: Optional[List[Dict[str, Any]]] = None) -> str:
        """コンテキストを用いてLLMで回答を生成"""
        try:
            # コンテキストを結合
            context_text = "\n\n".join(contexts)
            
            # プロンプトの構築
            prompt = f"""以下は、ユーザーの質問に関連するマニュアルからの情報です：

{context_text}

ユーザーの質問: {query}

上記の情報に基づいて、ユーザーの質問に明確に答えてください。マニュアルに記載されている情報のみを使用し、情報がない場合はその旨を伝えてください。"""
            
            # Ollamaリクエストの準備
            messages = []
            
            # 履歴がある場合は追加
            if history:
                for msg in history:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
            # 最後にユーザーの質問を追加
            messages.append({
                "role": "user", 
                "content": prompt
            })
            
            # Ollamaにリクエスト送信
            response = requests.post(
                f"{self.api_base}/chat",
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "model": self.model,
                    "messages": messages,
                    "stream": False
                })
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "回答を生成できませんでした。")
            else:
                self.logger.error(f"Ollamaエラー: {response.status_code} - {response.text}")
                return "LLMからの回答取得中にエラーが発生しました。"
        
        except Exception as e:
            self.logger.error(f"回答生成中にエラーが発生しました: {str(e)}")
            return "回答生成中にエラーが発生しました。"