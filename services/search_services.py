import os
import glob
import streamlit as st
from PIL import Image
from typing import List, Dict, Any
from tqdm import tqdm
from qdrant_client.models import PointStruct
import logging

from config.settings import Config
from utils.qdrant_helper import qdrant_helper
from utils.clip_helper import clip_helper
from endpoints.api_endpoints import api_client
from utils.helpers import load_image_from_path

logger = logging.getLogger(__name__)

class SearchService:
    """Service for handling different types of image searches"""
    
    def __init__(self):
        self.is_indexed = False
    
    @st.cache_resource
    def build_image_index(_self, force_rebuild: bool = False) -> bool:
        """Build image index in Qdrant"""
        if _self.is_indexed and not force_rebuild:
            return True
            
        try:
            # Create collection
            if not qdrant_helper.create_collection():
                return False
            
            # Get all image paths
            image_paths = _self._get_all_image_paths()
            if not image_paths:
                logger.warning("No images found in image store")
                return False
            
            # Process images and create embeddings
            points = []
            
            for idx, path in enumerate(tqdm(image_paths, desc="Processing images")):
                try:
                    image = load_image_from_path(path)
                    if image:
                        embedding = clip_helper.get_image_embedding(image)
                        points.append(
                            PointStruct(
                                id=idx,
                                vector=embedding.tolist(),
                                payload={"path": path, "index": idx}
                            )
                        )
                except Exception as e:
                    logger.warning(f"Skipping image {path}: {e}")
                    continue
            
            if points:
                success = qdrant_helper.upsert_points(points)
                if success:
                    _self.is_indexed = True
                    logger.info(f"Successfully indexed {len(points)} images")
                return success
            return False
            
        except Exception as e:
            logger.error(f"Error building image index: {e}")
            return False
    
    def search_by_feature(self, query: str) -> List[str]:
        """Search images by text query using CLIP"""
        try:
            if not self.is_indexed:
                if not self.build_image_index():
                    return []
            
            # Get text embedding
            text_embedding = clip_helper.get_text_embedding(query)
            
            # Search in Qdrant
            results = qdrant_helper.search_vectors(
                query_vector=text_embedding.tolist(),
                # limit=Config.DEFAULT_TOP_K,
                score_threshold=Config.SIMILARITY_THRESHOLD
            )
            
            return [result["payload"]["path"] for result in results]
        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return []
    
    def search_by_image(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Search similar images using image query"""
        try:
            if not self.is_indexed:
                if not self.build_image_index():
                    return []
            
            # Get image embedding
            image_embedding = clip_helper.get_image_embedding(image)
            
            # Search in Qdrant
            result = qdrant_helper.query_points(
                query=image_embedding.tolist(),
                limit=Config.IMAGE_TOP_K,
                score_threshold=Config.IMAGE_SIMILARITY_THRESHOLD,
                with_payload=True
            )
            
            if result and result.points:
                return [{
                    "path": point.payload.get("path"),
                    "score": point.score
                } for point in result.points]
            return []
        except Exception as e:
            logger.error(f"Error in image search: {e}")
            return []
    
    def search_by_metadata(self, query: str) -> List[str]:
        """Search images by metadata using external API"""
        return api_client.search_by_metadata(query)
    
    def hybrid_search(self, query: str) -> List[str]:
        """Combine metadata and feature-based search"""
        try:
            # Get results from metadata search
            metadata_results = self.search_by_metadata(query)
            
            if not metadata_results:
                return []
            
            # Load images and compare with query
            valid_images = []
            valid_paths = []
            
            for path in metadata_results:
                image = load_image_from_path(path)
                if image:
                    valid_images.append(image)
                    valid_paths.append(path)
            
            if not valid_images:
                return metadata_results
            
            # Use CLIP to rank images by relevance to query
            top_indices = clip_helper.compare_images_with_text(valid_images, query)
            
            return [valid_paths[idx] for idx in top_indices]
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def _get_all_image_paths(self) -> List[str]:
        """Get all image paths from image store"""
        folder_path = Config.IMAGE_STORE_PATH
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return []
        
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        image_paths = []
        
        for pattern in patterns:
            image_paths.extend(glob.glob(os.path.join(folder_path, pattern)))
        
        return image_paths


# Global instance
search_service = SearchService()
