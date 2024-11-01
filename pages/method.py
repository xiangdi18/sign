import streamlit as st

# Add custom HTML and CSS to style the caption 
st.markdown( """ <style> .caption { font-size: 20px; font-family: Arial, sans-serif; color: blue; } </style> """, unsafe_allow_html=True) 




st.title("Methodology (Flow Chart)")
st.image("method.jpg")
st.markdown('<div class="caption">Signature verfication flow</div>', unsafe_allow_html=True)

st.write(
"""
Step 1:

User requires to upload 2 document, the actual document for verification and 2nd the authorised signature for verification.

User uploads pdf or jpg document which contains the signature for verification, streamlit built-in checks to ensure only pdf or jpg document is allowed to upload. 

For signature, we only allow jpg to ensure that 1 signature is uploaded only. 

For the sake of this POC, we allow user to upload signature for verification. For actual production use case, the authorised signature will be uploaded/saved privately and users only need to upload the document for check.


"""
)


