import streamlit as st

# --- Page Setup ---
homepage = st.Page(
    page="pages/homepage.py",
    title='CarVision',
    icon='🚗',
    description='This is a simple web app to recognize cars.'
)

main_page = st.Page(
    page="pages/main.py",
    title='Image Prediction',
    icon='🔍',
    description='This is a simple web app to recognize cars.'
)

history_page = st.Page(
    page="pages/history.py",
    title='Prediction History',
    icon='📚',
    description='View the history of your predictions.'
)

# --- Navigation ---
pg st.navigation(pages=[main_page, history_page])

pg.run

#Homepage
st.title('CarVision')

