import streamlit as st
from pages.testv5 import main_page

st.set_page_config(page_title="AI Bootcamp Capstone Project", page_icon="🚀", layout="wide")

st.expander(
"""

IMPORTANT NOTICE: This web application is developed as a proof-of-concept prototype. The information provided here is NOT intended for actual usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.

Always consult with qualified professionals for accurate and personalized advice.

"""

)

def main_page():
    st.title("Main Page")
    st.write("Welcome to the main page of my app.")

def page_2():
    st.title("Page 2")
    st.write("This is the second page.")

def page_3():
    st.title("Page 3")
    st.write("This is the third page.")

page_names_to_funcs = {
    "Main Page": main_page,
    "Page 2": page_2,
    "Page 3": page_3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
