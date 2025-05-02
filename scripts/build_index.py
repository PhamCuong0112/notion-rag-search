import argparse
import logging
import os
from tqdm import tqdm

from app.core.config import get_settings
from app.core.notion import NotionAPI
from app.rag.embedding import TextProcessor
from app.rag.vector_store import VectorStore

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def build_index():
    """Notionページからインデックスを構築"""
    settings = get_settings()
    
    # Notionクライアント
    notion = NotionAPI()
    
    # テキスト処理
    text_processor = TextProcessor()
    
    # ベクトルストア
    vector_store = VectorStore()
    
    # データベースからページを取得
    logger.info(f"データベース {settings.notion_database_id} からページ一覧を取得しています...")
    pages = notion.get_database_pages()
    
    if not pages:
        logger.error("ページが見つかりませんでした。Notion APIトークンとデータベースIDを確認してください。")
        return
    
    logger.info(f"{len(pages)}個のページが見つかりました。処理を開始します...")
    
    # 各ページのコンテンツを取得して処理
    total_chunks = 0
    for page in tqdm(pages, desc="ページ処理中"):
        page_id = page["id"]
        
        # ページのタイトルを取得
        title = "不明なページ"
        try:
            title_property = page.get("properties", {}).get("title", {})
            if "title" in title_property:
                title = title_property["title"][0]["plain_text"]
            elif "Name" in page.get("properties", {}):
                name_property = page["properties"]["Name"]
                if "title" in name_property:
                    title = name_property["title"][0]["plain_text"]
        except (KeyError, IndexError):
            pass
        
        # ページURLの構築
        page_url = f"https://notion.so/{page_id.replace('-', '')}"
        
        # ページのコンテンツを取得
        blocks = notion.get_page_content(page_id)
        text = notion.extract_text_from_blocks(blocks)
        
        if not text:
            logger.warning(f"ページ '{title}' にテキストコンテンツがありません。スキップします。")
            continue
        
        # メタデータの設定
        metadata = {
            "page_id": page_id,
            "title": title,
            "url": page_url
        }
        
        # テキストを分割
        chunks = text_processor.split_text(text, metadata)
        
        if not chunks:
            logger.warning(f"ページ '{title}' のチャンク分割に失敗しました。スキップします。")
            continue
        
        # チャンクからテキストと埋め込みを抽出
        texts = [chunk["content"] for chunk in chunks]
        embeddings = text_processor.create_embeddings(texts)
        
        if not embeddings:
            logger.warning(f"ページ '{title}' の埋め込み生成に失敗しました。スキップします。")
            continue
        
        # ベクトルストアに追加
        vector_store.add_documents(chunks, embeddings)
        total_chunks += len(chunks)
    
    # ベクトルストアを保存
    if total_chunks > 0:
        if vector_store.save():
            logger.info(f"インデックスを構築しました。{len(pages)}ページから{total_chunks}個のチャンクがインデックス化されました。")
        else:
            logger.error("インデックスの保存に失敗しました。")
    else:
        logger.error("インデックス化するコンテンツがありませんでした。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Notionページからインデックスを構築するスクリプト")
    parser.add_argument("--force", action="store_true", help="既存のインデックスを上書きする")
    args = parser.parse_args()
    
    settings = get_settings()
    index_path = f"{settings.vector_store_path}/index.faiss"
    
    # インデックスが存在するかチェック
    if os.path.exists(index_path) and not args.force:
        logger.warning(f"インデックスファイル {index_path} が既に存在します。上書きする場合は --force オプションを使用してください。")
    else:
        build_index()