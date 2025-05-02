from typing import List, Dict, Any, Tuple
import logging

from app.core.config import get_settings
from app.rag.embedding import TextProcessor
from app.rag.vector_store import VectorStore

class RAGOrchestrator:
    def __init__(self):
        settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self.text_processor = TextProcessor()
        self.vector_store = VectorStore()
        self.top_k = settings.top_k
        
        # ベクトルストアの読み込み
        if not self.vector_store.load():
            self.logger.warning("ベクトルストアの読み込みに失敗しました。インデックスが構築されていることを確認してください。")
    
    def retrieve(self, query: str) -> Tuple[List[str], List[str]]:
        """クエリに関連するコンテキストを検索"""
        try:
            # クエリの埋め込みしてベクトル生成
            query_embedding = self.text_processor.embed_query(query)
            
            # 類似検索
            docs, _ = self.vector_store.similarity_search(query_embedding, k=self.top_k)
            
            # 結果の整形
            contexts = []
            sources = []
            
            for doc in docs:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                
                contexts.append(content)
                
                # ソース情報があれば追加
                page_title = metadata.get("title", "不明なページ")
                page_url = metadata.get("url", "")
                if page_url:
                    sources.append(f"{page_title} ({page_url})")
                else:
                    sources.append(page_title)
            
            return contexts, sources
        except Exception as e:
            self.logger.error(f"検索中にエラーが発生しました: {str(e)}")
            return [], []