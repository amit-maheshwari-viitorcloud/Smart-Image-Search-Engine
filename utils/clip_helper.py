import torch
import logging
import numpy as np
import streamlit as st
from typing import List, Tuple, Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from config.settings import Config

logger = logging.getLogger(__name__)

class CLIPHelper:
    """Helper class for CLIP model operations"""
    
    def __init__(self):
        self.model_id = Config.CLIP_MODEL_ID
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.processor, self.tokenizer = self.load_model()

    @staticmethod
    @st.cache_resource
    def load_model() -> Tuple[CLIPModel, CLIPProcessor, CLIPTokenizer]:
        """Load CLIP model, processor, and tokenizer with caching"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL_ID)
            tokenizer = CLIPTokenizer.from_pretrained(Config.CLIP_MODEL_ID)
            model = CLIPModel.from_pretrained(Config.CLIP_MODEL_ID).to(device)
            logger.info(f"CLIP model loaded successfully on {device}")
            return model, processor, tokenizer
        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
            raise
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding using CLIP"""
        try:
            text_inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
            with torch.no_grad():
                text_emb = self.model.get_text_features(**text_inputs)
            text_emb = text_emb.cpu().numpy().astype("float32")
            text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
            return text_emb[0]
        except Exception as e:
            logger.error(f"Error getting text embedding: {e}")
            return np.zeros((Config.EMBEDDING_DIM,), dtype="float32")

    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Get image embedding using CLIP"""
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_emb = self.model.get_image_features(**inputs)
            image_emb = image_emb.squeeze().cpu().numpy()
            image_emb = image_emb / np.linalg.norm(image_emb)
            return image_emb.astype("float32")
        except Exception as e:
            logger.error(f"Error getting image embedding: {e}")
            return np.zeros((Config.EMBEDDING_DIM,), dtype="float32")
    

    def compare_images_with_text(self, images: List[Image.Image], text: str) -> List[int]:
        """Compare multiple images with text query and return top matches"""
        try:
            inputs = self.processor(
                text=[text],
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=0).cpu().numpy().flatten()
            # top_indices = np.argsort(probs)[::-1][:Config.DEFAULT_TOP_K]
            top_indices = np.argsort(probs)[::-1]
            return top_indices.tolist()
        except Exception as e:
            logger.error(f"Error comparing images with text: {e}")
            return []

# Global instance
clip_helper = CLIPHelper()
