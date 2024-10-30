from PIL import Image
import streamlit as st

# Add custom HTML and CSS to style the caption 
st.markdown( """ <style> .caption { font-size: 20px; font-family: Arial, sans-serif; color: blue; } </style> """, unsafe_allow_html=True) 

image = Image.open("method.jpg")


st.title("Methodology (Flow Chart)")
st.image(Image)
st.markdown('<div class="caption">Signature verfication flow</div>', unsafe_allow_html=True)



