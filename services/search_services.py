import re
import os
import glob
import json
import logging
import streamlit as st
from PIL import Image
from typing import List, Dict, Any
from tqdm import tqdm
from datetime import datetime
from qdrant_client.models import PointStruct
from langchain_groq import ChatGroq

from config.settings import Config
from utils.qdrant_helper import qdrant_helper
from utils.clip_helper import clip_helper
from endpoints.api_endpoints import api_client
from utils.helpers import load_image_from_path
from agents.prompts import metadata_system_prompt


logger = logging.getLogger(__name__)

class SearchService:
    """Service for handling different types of image searches"""
    
    def __init__(self):
        self.is_indexed = False
    
    def get_single_range(self, text) -> str:
        current_year = datetime.now().year
        text = text.lower().strip()
        years = list(map(int, re.findall(r'\d{4}', text)))
        
        if 'after' in text and years:
            return years[0], current_year
        elif 'before' in text and years:
            start_year = 0
            return start_year, years[0]
        elif ';' in text or ',' in text:
            years = list(map(int, re.findall(r'\d{4}', text)))
            if len(years) >= 2:
                return min(years), max(years)
        elif len(years) == 2:
            return years[0], years[1]
        elif len(years) == 1:
            return years[0], years[0]
        
        return None, None


    def safe_lower(self, val):
        return val.lower() if isinstance(val, str) else "unknown"


    def store_sample_metadata(self) -> List[str]:
        points = []

        # from api_sample_data import sample_data
        search_query = ["Maharaja", "Mountains", "Tribal Art of India", "Photographs", "Baua Devi", "Flower", "Fruit", "Ancient Artwork", "Colonial period", "Saint", "Fashion", "British Rule", "Car"]

        logger.info("Initiated - Data Injection")
        for query in search_query:
            response = api_client.search_by_api(query).json()        
            data_dict = response["results"]["data"]
            logger.info(f"Fetching Data...")
            for idx, data in enumerate(data_dict):
                try:
                    period_start, period_end = self.get_single_range(data.get("period", "unknown"))
                    artists = data.get("artists", [])
                    artist_bios = [self.safe_lower(artist.get("bio") or "unknown") for artist in artists]
                    artist_names = [self.safe_lower(artist.get("name") or "unknown") for artist in artists]


                    dict_data = {
                        "id": data.get("id", "unknown"),
                        "date": data.get("date", "unknown"),
                        "accession_number": data.get("accession_number", "unknown"),
                        "medium": self.safe_lower(data.get("medium") or "unknown"),
                        "dimensions": data.get("dimensions", "unknown"),
                        "status": data.get("status", "unknown"),
                        "public_access": data.get("public_access", "unknown"),
                        "path": data.get("primary_image", "unknown"),
                        "instance_id": data.get("instance_id", "unknown"),
                        "title": data.get("title", "unknown"),
                        "department_id": data.get("department_id", "unknown"),
                        "department": self.safe_lower(data.get("department") or "unknown"),
                        "period_start": period_start,
                        "period_end": period_end,
                        "signed": data.get("signed", "unknown"),
                        "keywords": data.get("keywords", "unknown"),
                        "condition": data.get("condition", "unknown"),
                        "inscribed": data.get("inscribed", "unknown"),
                        "paper_support": self.safe_lower(data.get("paper_support") or "unknown"),
                        "attributes": data.get("attributes", "unknown"),
                        # "artist_bio": artist_bios,
                        "artist_bio": " | ".join(artist_bios) if artist_bios else "unknown",
                        # "artist_name": artist_names,
                        "artist_name": " | ".join(artist_names) if artist_names else "unknown"
                    }

                    image = load_image_from_path(data["primary_image"])
                    if image:
                        embedding = clip_helper.get_image_embedding(image)
                        points.append(
                            PointStruct(
                                id=data["id"],
                                vector=embedding.tolist(),
                                payload=dict_data
                            )
                        )
                except Exception as e:
                    logger.warning(f"Skipping data : {e}")
    
        logger.info("Completed - Data Injection")
        return points


    @st.cache_resource
    def build_image_index(_self, force_rebuild: bool = False) -> bool:
        """Build image index in Qdrant"""
        if _self.is_indexed and not force_rebuild:
            return True
            
        try:
            # Create collection
            if not qdrant_helper.create_collection():
                return False
            
            ##################################
            ## Store sample metadata points ##
            ##################################
            points = _self.store_sample_metadata()

            # #######################################
            # # Store data from image_store folder ##
            # #######################################
            # image_paths = _self._get_all_image_paths()
            # if not image_paths:
            #     logger.warning("No images found in image store")
            #     return False
            
            # points = []

            # # Process images and create embeddings
            # for idx, path in enumerate(tqdm(image_paths, desc="Processing images")):
            #     try:
            #         image = load_image_from_path(path)
            #         if image:
            #             embedding = clip_helper.get_image_embedding(image)
            #             points.append(
            #                 PointStruct(
            #                     id=idx,
            #                     vector=embedding.tolist(),
            #                     payload={"path": path, "index": idx}
            #                 )
            #             )
            #     except Exception as e:
            #         logger.warning(f"Skipping image {path}: {e}")
            #         continue
            
            ##################### Save data from the image_store folder till here #########################

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
            
            text_embedding = clip_helper.get_text_embedding(query)
            
            results = qdrant_helper.search_vectors(
                query_vector=text_embedding.tolist(),
                limit=Config.DEFAULT_TOP_K,
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
            
            image_embedding = clip_helper.get_image_embedding(image)
            
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

    
    def search_by_api(self, query: str) -> List[str]:
        """Search images by metadata using external API"""
        response = api_client.search_by_api(query)
        if response is None:
            return []
        elif response.status_code != 200:
            logger.error(f"API search failed with status code: {response.status_code}")
            return []
        else:
            data = response.json().get('results', {}).get('data', [])
            img_links = [item.get('primary_image') for item in data if item.get('primary_image')]
            
            logger.info(f"Found {len(img_links)} images for query: {query}")
            return img_links
 

    def initialize_llm(self):
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0,
            api_key=Config.GROQ_API_KEY
        )
        return llm


    def create_metadata(self, query: str) -> List[str]:
        messages = [
            ("system", metadata_system_prompt),
            ("human", query),
        ]
        llm = self.initialize_llm()    
        response = llm.invoke(messages).content
        result = json.loads(response)
        return result


    def search_by_metadata(self, query: str) -> List[str]:
        """Search images by metadata using external API"""        
        try:
            metadata_json = self.create_metadata(query)
            if not self.is_indexed:
                if not self.build_image_index():
                    return []

            text_embedding = clip_helper.get_text_embedding(query)

            results = qdrant_helper.metadata_based_searching(
                # query=query,
                query_vector=text_embedding.tolist(),
                metadata_json=metadata_json,
                limit=Config.IMAGE_TOP_K
            )

            return [result["payload"]["path"] for result in results]
        except Exception as e:
            logger.error(f"Error in metadata search: {e}")
            return []


    def hybrid_search(self, query: str) -> List[str]:
        """Combine metadata and feature-based search"""
        try:
            # metadata_results = self.search_by_api(query)            # For API based searching
            metadata_results = self.search_by_metadata(query)     # For Metadata based searching
            
            if not metadata_results:
                return []
            
            valid_images = []
            valid_paths = []
            
            for path in metadata_results:
                image = load_image_from_path(path)
                if image:
                    valid_images.append(image)
                    valid_paths.append(path)
            
            if not valid_images:
                return metadata_results
            
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


search_service = SearchService()
