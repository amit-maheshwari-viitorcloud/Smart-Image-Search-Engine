import streamlit as st
from .helpers import load_image_from_path

def show_results(image_paths):
    if not image_paths:
        st.warning("No results found.")
        return

    st.subheader(f"Results ({len(image_paths)})")
    cols_per_row = 4
    rows = (len(image_paths) + cols_per_row - 1) // cols_per_row

    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            idx = row * cols_per_row + col_idx
            if idx < len(image_paths):
                with cols[col_idx]:
                    img_path = image_paths[idx]
                    img = load_image_from_path(img_path)
                    if img:
                        st.image(img, caption=f"Result {idx + 1}", use_column_width=True)
                    # st.caption(str(img_path))
