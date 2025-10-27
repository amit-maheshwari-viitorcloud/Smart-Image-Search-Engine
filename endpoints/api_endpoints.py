import requests
import logging
from typing import Optional, List, Dict, Any
from config.settings import Config

logger = logging.getLogger(__name__)

class APIClient:
    """Client for external API operations"""
    
    def __init__(self):
        self.oauth_token = None
        
    def get_oauth_token(self) -> Optional[str]:
        """Get OAuth token for API authentication"""
        try:
            headers = {"Content-Type": "application/json"}
            payload = {
                "grant_type": "client_credentials",
                "client": Config.CUMULUS_CLIENT_ID
            }
            
            response = requests.post(Config.OAUTH_URL, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            token = response.json().get("access_token")
            if token:
                self.oauth_token = token
                logger.info("OAuth token obtained successfully")
            return token
        except Exception as e:
            logger.error(f"Error getting OAuth token: {e}")
            return None
    
    def search_by_api(self, search_keyword: str) -> List[str]:
        """Search images using metadata API"""
        try:
            if not self.oauth_token:
                self.get_oauth_token()
                
            if not self.oauth_token:
                return []
            
            params = {
                "q": search_keyword,
                "key": Config.CUMULUS_API_KEY
            }
            
            headers = {"Authorization": f"Bearer {self.oauth_token}"}
            
            response = requests.get(
                Config.SEARCH_API_URL, 
                headers=headers, 
                params=params, 
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json().get('results', {}).get('data', [])
            img_links = [item.get('primary_image') for item in data if item.get('primary_image')]
            
            logger.info(f"Found {len(img_links)} images for query: {search_keyword}")
            return img_links
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []

# Global instance
api_client = APIClient()
