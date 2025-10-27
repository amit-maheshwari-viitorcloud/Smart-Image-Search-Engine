import logging
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, Range, MatchValue
from qdrant_client.models import VectorParams, Distance, PointStruct, QueryResponse
from config.settings import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantHelper:
    """Helper class for Qdrant operations"""
    
    def __init__(self):
        self.client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)
        self.collection_name = Config.COLLECTION_NAME
        self.embedding_dim = Config.EMBEDDING_DIM
        
    def create_collection(self) -> bool:
        """Create Qdrant collection if it doesn't exist"""
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim, 
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection '{self.collection_name}' created successfully")
                return True
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return True
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    def upsert_points(self, points: List[PointStruct]) -> bool:
        """Insert or update points in the collection"""
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Successfully upserted {len(points)} points")
            return True
        except Exception as e:
            logger.error(f"Error upserting points: {e}")
            return False
    
    def search_vectors(self, query_vector: List[float], limit: int = 5, 
                      score_threshold: float = None) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        try:
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query_vector,
                "limit": limit
            }
            
            if score_threshold:
                search_params["score_threshold"] = score_threshold
            
            hits = self.client.search(**search_params)
            
            results = []
            for hit in hits:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                })
            
            return results
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    
    def query_points(self, query: List[float], limit: int = 5, 
                    score_threshold: float = None, with_payload: bool = True) -> Optional[QueryResponse]:
        """Query points with advanced options"""
        try:
            query_params = {
                "collection_name": self.collection_name,
                "query": query,
                "limit": limit,
                "with_payload": with_payload
            }
            
            if score_threshold:
                query_params["score_threshold"] = score_threshold
            
            return self.client.query_points(**query_params)
        except Exception as e:
            logger.error(f"Error querying points: {e}")
            return None


    def metadata_based_searching(self, query_vector: List[float], metadata_json: str, limit: int = 5) -> List[str]:
        """Search images by metadata using external API"""
        key = list(metadata_json.keys())[0]
        value = metadata_json[key]

        search_filter = Filter(
            must=[
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            ]
        )

        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,  # Your query vector
            query_filter=search_filter,
            limit=limit
        )
       
        results = []
        for searches in search_results:
            results.append({
                "id": searches.id,
                "score": searches.score,
                "payload": searches.payload
            })
       
        return results



# Global instance
qdrant_helper = QdrantHelper()
