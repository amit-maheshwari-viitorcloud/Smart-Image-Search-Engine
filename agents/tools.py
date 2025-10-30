from langchain_core.tools import tool
from services.search_services import search_service


@tool
def search_by_feature(query: str):
    """Search images by text for queries about visual characteristics only."""
    return search_service.search_by_feature(query)

@tool
def search_by_metadata(query: str):
    """Search images for queries about metadata only (artist, title, date, etc)."""
    # return search_service.search_by_api(query)
    return search_service.search_by_metadata(query)

@tool
def search_hybrid(query: str):
    """Search images by both metadata and features."""
    return search_service.hybrid_search(query)

@tool
def random_search(query: str):
    """Handle gibberish, random, or meaningless queries."""
    return []

tools = [search_by_feature, search_by_metadata, search_hybrid, random_search]
