import os
import tempfile
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Smart Image Search (Agent)", page_icon="🔍", layout="wide")
st.title("🔍 Smart Image Search Engine")

from config.settings import Config
from utils.helpers import validate_image
from utils.ui_helpers import show_results
from services.search_services import search_service
from agents.agent_executor import initialize_agent, agent_search


@st.cache_resource
def initialize_app():
    """Initialize the application"""
    try:
        Config.validate_config()
        with st.spinner("Initializing search engine..."):
            if not search_service.is_indexed:
                success = search_service.build_image_index()
                if not success:
                    st.error("Failed to initialize search engine. Please check your image store.")
                    return False
        return True
    except Exception as e:
        st.error(f"Configuration error: {e}")
        return False


def process_image_search(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    try:
        query_image = Image.open(temp_path).convert("RGB")
        with st.spinner("Searching by image..."):
            results = search_service.search_by_image(query_image)
            show_results([r["path"] for r in results])
            if results:
                st.subheader("Similarity Scores")
                for i, r in enumerate(results):
                    st.write(f"Result {i+1}: {r['score']:.3f}")
    finally:
        os.unlink(temp_path)


# ---- Streamlit Interface ----
def main():
    """Main application function"""
    if not initialize_app():
        return
    
    executor = initialize_agent()

    query = st.text_input(
        "Enter your search (by description, artist, feature, etc):",
        placeholder="e.g. blue sky, Van Gogh, 19th-century Indian landscape"
    )

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if st.button("Search", type="primary"):
        if uploaded_file and validate_image(uploaded_file):
            st.image(uploaded_file, caption="Uploaded Image")
            process_image_search(uploaded_file)
        elif query:
            with st.spinner("Agent is analyzing and searching..."):
                image_paths = agent_search(executor, query)
                show_results(image_paths)
        else:
            st.error("Please enter a search query or upload an image.")


if __name__ == "__main__":
    main()

