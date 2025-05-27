import faiss
import numpy as np
import pickle
import os
import logging
from typing import List, Dict, Any, Tuple, Optional

from app.core.config import get_settings

class VectorStore:
    def __init__(self, embedding_size: Optional[int] = None):
        """
        ベクトルストアを初期化
        
        Args:
            embedding_size: 埋め込みベクトルの次元数（Noneの場合、最初の追加時に自動検出）
        """
        settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self.index = None
        self.embedding_size = embedding_size
        self.documents = []
        self.vector_store_path = settings.vector_store_path
    
    def _initialize_index(self, dimension: int) -> None:
        """
        指定された次元数でFAISSインデックスを初期化
        
        Args:
            dimension: 埋め込みベクトルの次元数
        """
        if self.index is None or self.embedding_size != dimension:
            self.embedding_size = dimension
            self.index = faiss.IndexFlatL2(dimension)
            self.logger.info(f"FAISSインデックスを次元数 {dimension} で初期化しました")
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """ドキュメントとその埋め込みをベクトルストアに追加"""
        try:
            if not documents or not embeddings:
                self.logger.warning("追加するドキュメントまたは埋め込みが空です")
                return
            
            # 長さチェック
            if len(documents) != len(embeddings):
                self.logger.error(f"ドキュメント数と埋め込み数が一致しません: documents={len(documents)}, embeddings={len(embeddings)}")
                return
            
            # 埋め込みをnumpy配列に変換
            embeddings_np = np.array(embeddings, dtype=np.float32)
            
            # 埋め込みの次元数を取得
            if embeddings_np.shape[0] > 0:
                embedding_dimension = embeddings_np.shape[1]
                
                # インデックスがまだ初期化されていない場合は初期化
                if self.index is None:
                    self._initialize_index(embedding_dimension)
                
                # 既存のインデックスと次元数が一致しない場合はエラー
                if self.embedding_size != embedding_dimension:
                    self.logger.error(f"埋め込みの次元数が一致しません: インデックス={self.embedding_size}, 埋め込み={embedding_dimension}")
                    return
                
                # FAISSインデックスにベクトルを追加
                self.index.add(embeddings_np)
                
                # 元のドキュメントを保存
                self.documents.extend(documents)
                self.logger.info(f"{len(documents)}個のドキュメントをベクトルストアに追加しました")
            else:
                self.logger.warning("追加する埋め込みが空です")
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"ドキュメント追加中にエラーが発生しました: {str(e)}\n{error_details}")
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> Tuple[List[Dict[str, Any]], List[float]]:
        """クエリ埋め込みに最も近いドキュメントを検索"""
        try:
            # インデックスが初期化されていない場合はエラー
            if self.index is None:
                self.logger.error("インデックスが初期化されていません")
                return [], []
            
            # ドキュメントが空の場合は空の結果を返す
            if not self.documents:
                self.logger.warning("ドキュメントが存在しないため検索できません")
                return [], []
            
            query_embedding_np = np.array([query_embedding], dtype=np.float32)
            
            # クエリ埋め込みの次元数がインデックスと一致するか確認
            if query_embedding_np.shape[1] != self.embedding_size:
                self.logger.error(f"クエリ埋め込みの次元数がインデックスと一致しません: クエリ={query_embedding_np.shape[1]}, インデックス={self.embedding_size}")
                return [], []
            
            # インデックスが空の場合は空の結果を返す
            if self.index.ntotal == 0:
                self.logger.warning("インデックスが空のため検索できません")
                return [], []
            
            # 検索実行
            distances, indices = self.index.search(query_embedding_np, min(k, self.index.ntotal))
            
            results = []
            for i, idx in enumerate(indices[0]):
                # FAISSは検索時に類似のものがない場合、-1を返すことがあるため、インデックスが有効かチェック
                if idx >= 0 and idx < len(self.documents):
                    results.append((self.documents[idx], float(distances[0][i])))
            
            # 距離でソート（最も近いものが先頭）
            results.sort(key=lambda x: x[1])
            
            # ドキュメントと距離を分離
            docs = [doc for doc, _ in results]
            distances = [dist for _, dist in results]
            
            return docs, distances
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"検索中にエラーが発生しました: {str(e)}\n{error_details}")
            return [], []
    
    def save(self) -> bool:
        """ベクトルストアを保存"""
        try:
            # インデックスが初期化されていない場合はエラー
            if self.index is None:
                self.logger.error("インデックスが初期化されていないため保存できません")
                return False
            
            if not self.documents or self.index.ntotal == 0:
                self.logger.warning("保存するドキュメントまたはインデックスが空です")
                # 空でも保存を試みる
            
            os.makedirs(self.vector_store_path, exist_ok=True)
            
            # ドキュメントを保存
            with open(f"{self.vector_store_path}/documents.pkl", "wb") as f:
                pickle.dump(self.documents, f)
            
            # FAISSインデックスを保存
            faiss.write_index(self.index, f"{self.vector_store_path}/index.faiss")
            
            self.logger.info(f"ベクトルストアを {self.vector_store_path} に保存しました（ドキュメント数: {len(self.documents)}、ベクトル数: {self.index.ntotal}）")
            return True
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"ベクトルストア保存中にエラーが発生しました: {str(e)}\n{error_details}")
            return False
    
    def load(self) -> bool:
        """ベクトルストアを読み込み"""
        try:
            documents_path = f"{self.vector_store_path}/documents.pkl"
            index_path = f"{self.vector_store_path}/index.faiss"
            
            # ファイルが存在するか確認
            if not os.path.exists(documents_path):
                self.logger.error(f"ドキュメントファイルが見つかりません: {documents_path}")
                return False
                
            if not os.path.exists(index_path):
                self.logger.error(f"インデックスファイルが見つかりません: {index_path}")
                return False
            
            # ドキュメントを読み込み
            try:
                with open(documents_path, "rb") as f:
                    self.documents = pickle.load(f)
                self.logger.info(f"ドキュメントファイルを読み込みました: {len(self.documents)}個のドキュメント")
            except Exception as e:
                self.logger.error(f"ドキュメントファイル読み込み中にエラーが発生しました: {str(e)}")
                return False
            
            # FAISSインデックスを読み込み
            try:
                self.index = faiss.read_index(index_path)
                self.embedding_size = self.index.d  # インデックスから次元数を取得
                self.logger.info(f"FAISSインデックスを読み込みました: {self.index.ntotal}個のベクトル、次元数: {self.embedding_size}")
            except Exception as e:
                self.logger.error(f"FAISSインデックス読み込み中にエラーが発生しました: {str(e)}")
                return False
            
            self.logger.info(f"ベクトルストアを {self.vector_store_path} から読み込みました（{len(self.documents)}個のドキュメント）")
            return True
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"ベクトルストア読み込み中にエラーが発生しました: {str(e)}\n{error_details}")
            return False
    
    def get_index_size(self) -> int:
        """インデックスのサイズ（ベクトル数）を返す"""
        if self.index is not None:
            return self.index.ntotal
        return 0
    
    def get_index_info(self) -> Dict[str, Any]:
        """インデックスの情報を返す"""
        return {
            "vector_count": self.get_index_size(),
            "dimension": self.embedding_size if self.embedding_size is not None else "未初期化",
            "documents_count": len(self.documents)
        }