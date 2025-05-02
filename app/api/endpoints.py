from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.core.config import get_settings
from app.rag.orchestrator import RAGOrchestrator
from app.llm.ollama import OllamaClient

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """チャットエンドポイント - ユーザーの質問に回答"""
    settings = get_settings()
    
    # RAGオーケストレーターの初期化
    rag = RAGOrchestrator()
    
    # 関連コンテキストを取得
    contexts, sources = rag.retrieve(request.query)
    
    # LLMクライアントの初期化
    llm = OllamaClient(model_name=settings.llm_model)
    
    # 回答の生成
    answer = llm.generate_response(request.query, contexts, history=request.history)
    
    return ChatResponse(answer=answer, sources=sources)