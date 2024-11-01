import streamlit as st

# CSS for tooltips
st.markdown("""
    <style>
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: black;
        color: #fff;
        text-align: center;
        border-radius: 5px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

# Tooltip content
def create_tooltip(text, tooltip):
    return f'<span class="tooltip">{text}<span class="tooltiptext">{tooltip}</span></span>'

# Example functions and their documentation
functions_docs = {
    "Function 1": "This function does X, Y, and Z.",
    "Function 2": "This function performs A, B, and C.",
    "Function 3": "This function is responsible for P, Q, and R."
}

# Display functions with mouseover tooltips
st.sidebar.title("Select a Function")
for function_name, doc in functions_docs.items():
    st.sidebar.markdown(create_tooltip(function_name, doc), unsafe_allow_html=True)

# Main content
st.write("Hover over the function names in the sidebar to see their documentation.")
