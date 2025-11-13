import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Qdrant Configuration
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "image_embeddings")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSION_SIZE", "512"))
    
    # API Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    CUMULUS_CLIENT_ID = os.getenv("CUMULUS_CLIENT_ID", "CUMULUS-IDCCG-PUB")
    CUMULUS_API_KEY = os.getenv("CUMULUS_API_KEY", "41204aed-89d7-4a45-9455-976ac475a8ab")
    
    # Model Configuration
    CLIP_MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
    IMAGE_STORE_PATH = os.getenv("IMAGE_STORE_PATH", "image_store")
    
    # Search Configuration
    DEFAULT_TOP_K = 2000
    IMAGE_TOP_K = 10000
    SIMILARITY_THRESHOLD = 0.2
    IMAGE_SIMILARITY_THRESHOLD = 0.75
    
    # URLs
    OAUTH_URL = "https://accounts.cumulus.co.in/oauth/token"
    SEARCH_API_URL = "https://srcapi.cumulus.co.in/api/public_hook/v1/artwork/"
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration"""
        required_vars = []
        if not cls.GROQ_API_KEY:
            required_vars.append("GROQ_API_KEY")
        if required_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(required_vars)}")
        return True
