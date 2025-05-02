from notion_client import Client
from typing import List, Dict, Any, Optional
import logging

from app.core.config import get_settings

class NotionAPI:
    def __init__(self, token: Optional[str] = None):
        settings = get_settings()
        self.client = Client(auth=token or settings.notion_token)
        self.logger = logging.getLogger(__name__)
    
    def get_page_content(self, page_id: str) -> Dict[str, Any]:
        """ページの内容を取得"""
        try:
            return self.client.blocks.children.list(block_id=page_id)
        except Exception as e:
            self.logger.error(f"ページ内容の取得中にエラーが発生しました: {str(e)}")
            return {"results": []}
    
    def get_parent_page_content(self) -> List[Dict[str, Any]]:
        """親ページとその子ページの内容を再帰的に取得"""
        settings = get_settings()
        page_id = settings.notion_page_id
        
        if not page_id:
            self.logger.error("親ページIDが設定されていません")
            return []
        
        try:
            result = []
            # 再帰的に親ページとすべての子ページを処理
            self._process_page_recursive(page_id, result)
            return result
            
        except Exception as e:
            self.logger.error(f"親ページの取得中にエラーが発生しました: {str(e)}")
            return []
    
    def _process_page_recursive(self, page_id: str, result: List[Dict[str, Any]]) -> None:
        """ページとその子ページを再帰的に処理する"""
        try:
            # ページの基本情報を取得
            page_info = self.client.pages.retrieve(page_id=page_id)
            
            # ページのコンテンツを取得
            blocks = self.get_page_content(page_id)
            
            # ページのタイトルを取得
            title = "不明なページ"
            try:
                if "properties" in page_info and "title" in page_info["properties"]:
                    title_property = page_info["properties"]["title"]
                    title = title_property["title"][0]["plain_text"]
            except (KeyError, IndexError):
                pass
            
            # ページURLの構築
            # APIではハイフン付きで変えるので、URLを構築する際にハイフンを削除
            page_url = f"https://notion.so/{page_id.replace('-', '')}"
            
            # テキストを抽出
            text = self.extract_text_from_blocks(blocks)
            
            # 現在のページ情報を結果リストに追加
            result.append({
                "id": page_id,
                "title": title,
                "url": page_url,
                "content": text
            })
            
            # 子ページを再帰的に取得
            for block in blocks.get("results", []):
                if block.get("type") == "child_page":
                    child_page_id = block.get("id")
                    # 子ページを再帰的に処理
                    self._process_page_recursive(child_page_id, result)
                    
        except Exception as e:
            self.logger.error(f"ページID {page_id} の処理中にエラーが発生しました: {str(e)}")
    
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