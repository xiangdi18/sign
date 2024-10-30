import streamlit as st

# Add custom HTML and CSS to style the caption 
st.markdown( """ <style> .caption { font-size: 20px; font-family: Arial, sans-serif; color: blue; } </style> """, unsafe_allow_html=True) 




st.title("Methodology (Flow Chart)")
st.image("method.jpg")
st.markdown('<div class="caption">Signature verfication flow</div>', unsafe_allow_html=True)



