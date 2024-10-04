import streamlit as st
import pandas as pd
#import cv2
import matplotlib.pyplot as plt
import numpy as np
import tempfile 
#import os

def show_image(img):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

#Wand is a Python binding of ImageMagick, so you have to install it as well on local computer


from signature_detect.cropper import Cropper
from signature_detect.extractor import Extractor
from signature_detect.loader import Loader
from signature_detect.judger import Judger


st.title('Capstone Draft')

uploaded_file=""

uploaded_file = st.file_uploader("Upload a PDF to validate the signature.",type=['jpg','png'])





 

def check_sign(file_path):
    loader = Loader()

    #It returns a list of the masks. Each mask is a numpy 2 dimensions array. Its element's value is 0 or 255.
    mask = loader.get_masks(file_path)[0]

    st.image(mask)

    #Extractor reads a mask, labels the regions in the mask, and removes both small and big regions. We consider that the signature is a region of middle size.
    extractor = Extractor(min_area_size = 2,amplfier=15)

    labeled_mask = extractor.extract(mask)
    #st.write(labeled_mask)
    #st.write(np.unique(labeled_mask))


    #Cropper crops the regions in the labeled mask.

    cropper = Cropper(min_region_size=100)

    #The cropper finds the contours of regions in the labeled masks and crop them.
    results = cropper.run(labeled_mask)
    


    #Judger decides whether a region is a signature.

    judger = Judger()

    is_signed = False
    final_image=""

    for result in results.values():
        is_signed = judger.judge(result["cropped_mask"])
        st.image(result["cropped_mask"])
        if is_signed:
            final_image= result["cropped_mask"]
            break

    if not is_signed:

        st.write("no signature")

    else :
        st.write("This is the identified signature, please verify :")
        st.image(final_image)   


# Display uploaded image & save the image as the image uploaded is in memory.
if uploaded_file is not None:
    
    st.image(uploaded_file)
       
 # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
        
        # Define the path where the file will be saved
    file_path = os.path.join(temp_dir, uploaded_file.name)

        # Save the uploaded file to the temporary directory
    with open(file_path, "wb") as f:
       f.write(uploaded_file.getvalue())


        # Display the file path
    st.write(f"File saved at: " + file_path)

    check_sign(file_path)
