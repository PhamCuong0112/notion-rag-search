import faiss
import numpy as np
import pickle
import os
import logging
from typing import List, Dict, Any, Tuple

from app.core.config import get_settings

class VectorStore:
    def __init__(self, embedding_size: int = 768):
        settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self.index = faiss.IndexFlatL2(embedding_size)
        self.documents = []
        self.vector_store_path = settings.vector_store_path
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """ドキュメントとその埋め込みをベクトルストアに追加"""
        try:
            embeddings_np = np.array(embeddings, dtype=np.float32)

            # FAISSインデックスにベクトルを追加
            self.index.add(embeddings_np)

            # 元のドキュメントを保存
            self.documents.extend(documents)
            self.logger.info(f"{len(documents)}個のドキュメントをベクトルストアに追加しました")
        except Exception as e:
            self.logger.error(f"ドキュメント追加中にエラーが発生しました: {str(e)}")
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> Tuple[List[Dict[str, Any]], List[float]]:
        """クエリ埋め込みに最も近いドキュメントを検索"""
        try:
            query_embedding_np = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.index.search(query_embedding_np, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                # FAISSは検索時に類似のものがない場合、-1を返すことがあるため、インデックスが有効かチェック
                if idx >= 0 and idx < len(self.documents):
                    results.append((self.documents[idx], float(distances[0][i])))
            
            # 距離でソート（最も近いものが先頭）
            # results = [(document1, distance1), (document2, distance2), ...]
            results.sort(key=lambda x: x[1])
            
            # ドキュメントと距離を分離
            docs = [doc for doc, _ in results]
            distances = [dist for _, dist in results]
            
            return docs, distances
        except Exception as e:
            self.logger.error(f"検索中にエラーが発生しました: {str(e)}")
            return [], []
    
    def save(self) -> bool:
        """ベクトルストアを保存"""
        try:
            os.makedirs(self.vector_store_path, exist_ok=True)
            
            # ドキュメントを保存
            with open(f"{self.vector_store_path}/documents.pkl", "wb") as f:
                pickle.dump(self.documents, f)
            
            # FAISSインデックスを保存
            faiss.write_index(self.index, f"{self.vector_store_path}/index.faiss")
            
            self.logger.info(f"ベクトルストアを {self.vector_store_path} に保存しました")
            return True
        except Exception as e:
            self.logger.error(f"ベクトルストア保存中にエラーが発生しました: {str(e)}")
            return False
    
    def load(self) -> bool:
        """ベクトルストアを読み込み"""
        try:
            # ドキュメントを読み込み
            with open(f"{self.vector_store_path}/documents.pkl", "rb") as f:
                self.documents = pickle.load(f)
            
            # FAISSインデックスを読み込み
            self.index = faiss.read_index(f"{self.vector_store_path}/index.faiss")
            
            self.logger.info(f"ベクトルストアを {self.vector_store_path} から読み込みました（{len(self.documents)}個のドキュメント）")
            return True
        except Exception as e:
            self.logger.error(f"ベクトルストア読み込み中にエラーが発生しました: {str(e)}")
            return False