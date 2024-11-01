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
    "find_signature": "This function does X, Y, and Z.",
    "Function 2": "This function performs A, B, and C.",
    "Function 3": "This function is responsible for P, Q, and R."
}

# Display functions with mouseover tooltips
st.sidebar.title("Select a Function")
for function_name, doc in functions_docs.items():
    st.sidebar.markdown(create_tooltip(function_name, doc), unsafe_allow_html=True)

# Main content
st.write("Hover over the function names in the sidebar to see their documentation.")

"""
Step 1:

User requires to upload 2 document, the actual document for verification and 2nd the authorised signature for verification.

User uploads pdf or jpg document which contains the signature for verification, streamlit built-in checks to ensure only pdf or jpg document is allowed to upload. 

For signature, we only allow jpg to ensure that only 1 signature is uploaded. 

As streamlit stores documents inside memory. we would need to convert into jpg images first before we can perform any image manipulation.

jpgmaker() is used to convert any pdf document to jpg and stores into upload folder.

    It extracts the filename and appends page number to it.
    Finally, it returns the total pages , array of jpg location.


For the sake of this POC, we allow user to upload signature for verification. 

***For actual production use case, the authorised signature will be uploaded/saved privately and users only need to upload the document for check.


Step 2:

We create a find_signature() function that helps to identify the signature to the authorised signature.

Looping through each page of the document to the find_signature() to help to identify the signature area and returns the results to user at the end.

find_signature() consists of a few sub modules to help with the signature detection.

    template_matching() 
        # Function to dynamically scales the authorised image(signature) to find and crop the best matching region within a main image, returning this region if a match is found, or None otherwise.
            a) converts the uploaded image to grayscale and apply Gaussian Blur to reduce noise from the image.
            b) resize the authorised signature using scale granularity: 0.5 to 1.5 in increments of 0.05  (try using less granular but results are not as good hence make it smaller and more granular however will have performance impact)
            c) 
    This function dynamically scales a template image to find and crop the best matching region within a main image. If a match is found, it returns the region of interest (ROI); otherwise, it returns None. Args: main_image_path (str): Path to the main image file. template_image_path (str): Path to the template image file. min_match_quality (float, optional): Minimum match quality required to consider a match valid. Defaults to 0.8. Returns: numpy.ndarray: The cropped region of the main image where the template best matches, or None if no match is found. Raises: ValueError: If either the main image or the template image cannot be loaded.

Step 3:

Provides a summary on the signature found on pages xxx out of the whole document.

"""
# Example code to display
code = '''
def greet(name):
    return f"Hello, {name}!"

print(greet("World"))
'''

# Display the code snippet using st.code
st.code(code, language='python')
