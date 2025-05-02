from notion_client import Client
from typing import List, Dict, Any, Optional
import logging

from app.core.config import get_settings

class NotionAPI:
    def __init__(self, token: Optional[str] = None):
        settings = get_settings()
        self.client = Client(auth=token or settings.notion_token)
        self.logger = logging.getLogger(__name__)
    
    def get_database_pages(self, database_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """データベース内のすべてのページを取得"""
        settings = get_settings()
        db_id = database_id or settings.notion_database_id
        
        if not db_id:
            self.logger.error("データベースIDが設定されていません")
            return []
        
        try:
            response = self.client.databases.query(database_id=db_id)
            return response.get("results", [])
        except Exception as e:
            self.logger.error(f"データベースの取得中にエラーが発生しました: {str(e)}")
            return []
    
    def get_page_content(self, page_id: str) -> Dict[str, Any]:
        """ページの内容を取得"""
        try:
            return self.client.blocks.children.list(block_id=page_id)
        except Exception as e:
            self.logger.error(f"ページ内容の取得中にエラーが発生しました: {str(e)}")
            return {"results": []}
    
    def extract_text_from_blocks(self, blocks: Dict[str, Any]) -> str:
        """Notionブロックからテキストを抽出"""
        text = ""
        for block in blocks.get("results", []):
            block_type = block.get("type")
            
            if block_type == "paragraph":
                text += self._extract_text_from_rich_text(block.get("paragraph", {}).get("rich_text", []))
            elif block_type == "heading_1":
                text += "# " + self._extract_text_from_rich_text(block.get("heading_1", {}).get("rich_text", []))
            elif block_type == "heading_2":
                text += "## " + self._extract_text_from_rich_text(block.get("heading_2", {}).get("rich_text", []))
            elif block_type == "heading_3":
                text += "### " + self._extract_text_from_rich_text(block.get("heading_3", {}).get("rich_text", []))
            elif block_type == "bulleted_list_item":
                text += "• " + self._extract_text_from_rich_text(block.get("bulleted_list_item", {}).get("rich_text", []))
            elif block_type == "numbered_list_item":
                text += "1. " + self._extract_text_from_rich_text(block.get("numbered_list_item", {}).get("rich_text", []))
            elif block_type == "code":
                code_text = self._extract_text_from_rich_text(block.get("code", {}).get("rich_text", []))
                language = block.get("code", {}).get("language", "")
                text += f"```{language}\n{code_text}\n```"
            elif block_type == "quote":
                text += "> " + self._extract_text_from_rich_text(block.get("quote", {}).get("rich_text", []))
            
            text += "\n\n"
        
        return text.strip()
    
    def _extract_text_from_rich_text(self, rich_text: List[Dict[str, Any]]) -> str:
        """リッチテキスト配列からプレーンテキストを抽出"""
        return "".join([rt.get("plain_text", "") for rt in rich_text])