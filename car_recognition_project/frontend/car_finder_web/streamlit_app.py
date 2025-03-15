import streamlit as st

# --- Page Setup ---
homepage = st.Page(
    page="views/homepage.py",
    title='CarVision Home',
    icon='🚗',
    default=True
)

main_page = st.Page(
    page="views/main.py",
    title='Image Prediction',
    icon='🔍'
)

history_page = st.Page(
    page="views/history.py",
    title='Prediction History',
    icon='📚'
)

# --- Navigation/sidebar ---
pg = st.navigation(pages = [homepage, main_page, history_page])
pg.run()