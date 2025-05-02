import argparse
import logging

from app.rag.orchestrator import RAGOrchestrator
from app.llm.ollama import OllamaClient

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_rag_query(query: str, show_contexts: bool = False):
    """RAGシステムのテスト"""
    logger.info(f"クエリのテスト: {query}")
    
    # RAGオーケストレーターの初期化
    rag = RAGOrchestrator()
    
    # LLMクライアントの初期化
    llm = OllamaClient()
    
    # 関連コンテキストを取得
    contexts, sources = rag.retrieve(query)
    
    if not contexts:
        logger.error("関連コンテキストが見つかりませんでした。インデックスが正しく構築されているか確認してください。")
        return
    
    # コンテキストの表示
    if show_contexts:
        logger.info(f"{len(contexts)}個の関連コンテキストが見つかりました:")
        for i, (context, source) in enumerate(zip(contexts, sources), 1):
            logger.info(f"コンテキスト {i} (ソース: {source}):")
            logger.info(f"{context[:200]}...")
            logger.info("-" * 50)
    
    # 回答の生成
    logger.info("回答を生成しています...")
    response = llm.generate_response(query, contexts)
    
    # 結果の表示
    logger.info("回答:")
    print("\n" + "-" * 80)
    print(response)
    print("-" * 80)
    
    logger.info("ソース:")
    for i, source in enumerate(sources, 1):
        print(f"{i}. {source}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGシステムのテスト")
    parser.add_argument("query", type=str, help="テストするクエリ")
    parser.add_argument("--show-contexts", action="store_true", help="関連コンテキストを表示する")
    args = parser.parse_args()
    
    test_rag_query(args.query, args.show_contexts)