from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
import logging

from app.core.config import get_settings

class TextProcessor:
    def __init__(self):
        settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # テキスト分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # 埋め込みモデル
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
            self.logger.info(f"埋め込みモデル {settings.embedding_model} を読み込みました")
        except Exception as e:
            self.logger.error(f"埋め込みモデルの読み込み中にエラーが発生しました: {str(e)}")
            raise
    
    def split_text(self, text: str, metadata: dict = None) -> List[dict]:
        """テキストを分割してメタデータを追加"""
        try:
            chunks = self.text_splitter.split_text(text)
            
            # メタデータを各チャンクに追加
            chunks_with_metadata = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata["chunk_id"] = i
                chunks_with_metadata.append({
                    "content": chunk,
                    "metadata": chunk_metadata
                })
            
            return chunks_with_metadata
        except Exception as e:
            self.logger.error(f"テキスト分割中にエラーが発生しました: {str(e)}")
            return []
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """テキストの埋め込みベクトルを生成"""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            self.logger.error(f"埋め込み生成中にエラーが発生しました: {str(e)}")
            return []
    
    def embed_query(self, query: str) -> List[float]:
        """クエリの埋め込みベクトルを生成"""
        try:
            return self.embeddings.embed_query(query)
        except Exception as e:
            self.logger.error(f"クエリ埋め込み生成中にエラーが発生しました: {str(e)}")
            return []