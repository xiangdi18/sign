import streamlit as st
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import tempfile 
import io
import os
import fitz  # PyMuPDF
from PIL import Image
from skimage.metrics import structural_similarity as ssim
#Wand is a Python binding of ImageMagick, so you have to install it as well on local computer
from signature_detect.cropper import Cropper
from signature_detect.extractor import Extractor
from signature_detect.loader import Loader
from signature_detect.judger import Judger
from openai import OpenAI
import base64


client = OpenAI(
    api_key= st.secrets["OPENAI_API_KEY"],
    base_url="https://litellm.govtext.gov.sg/",
    default_headers={"user-agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/81.0"},
)


# Function to load and preprocess an image
#The final step normalizes the image tensor by subtracting the mean and dividing by the standard deviation for each channel (red, green, blue). 
#These values are standard for images used with models trained on the ImageNet dataset, ensuring consistency in the input data.

# Convert numpy array to PIL Image
def preprocess_image(np_image):
    
    if np_image.ndim == 2:  # Grayscale image
        pil_image = Image.fromarray(np.uint8(np_image), mode='L')
    elif np_image.ndim == 3 and np_image.shape[2] == 3:  # RGB image
        pil_image = Image.fromarray(np.uint8(np_image))
    else:
        raise ValueError("Unexpected image format")

    # Convert grayscale image to RGB if necessary
    if pil_image.mode == 'L':
        pil_image = pil_image.convert("RGB")
    
       
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(pil_image).unsqueeze(0)

# Function to extract features using ResNet
def extract_features(image):
    model = models.resnet50(pretrained=True)
    model.eval()
    with torch.no_grad():
        features = model(image)
    return features.flatten()

# Function to compare features using cosine similarity
def cosine_similarity(features1, features2):
    # Ensure features are 1D tensors
    if features1.dim() != 1:
        features1 = features1.flatten()
    if features2.dim() != 1:
        features2 = features2.flatten()

    cos_sim = torch.nn.functional.cosine_similarity(features1.unsqueeze(0), features2.unsqueeze(0))
    return cos_sim.item()



    return response.choices[0].message.content


# Function to describe an image using GPT-4 Vision
def find_image_with_gpt4(image,signature):
    response = client.chat.completions.create(
        messages=[
            {
                
                "role": "system", "content": "You will be provided with 2 images and try to find the signature that might be in any part of the first image. ",
                "role": "user", "content":[
                    {
                    "type": "text", 
                    "text": f"Try to find the 2nd image that is in a certain section of the first image, try rotating and scaling the 2nd image to try to find it location in the first image. Highlight the area in the first image and show it."
                    },
                    {"type": "image_url",
                    "image_url": {
                    "url":f"data:image/jpeg;base64,{image}" 
                    }
                    },
                     {
                    "type": "image_url",
                    "image_url": {
                        "url":f"data:image/jpeg;base64,{signature}" 
                        }                    
                    }
                ]
            }
        ],
            model="gpt-4o-mini-prd-gcc2-lb",)

    return response.choices[0].message.content

# Function to describe an image using GPT-4 Vision and try to find similiarity between 2 images.
def decide_gpt4(desc1,desc2):
    response = client.chat.completions.create(
        messages=[
            {
                
                "role": "system", "content": "You are an expert in signature recognization and with the ability to describle the signature with great detail.If unable to view the image, explain why you are unable to do so.",
                "role": "user", "content":[
                {
                    "type": "text", 
                    "text":
                    """
                    1.Line Quality: Look at the smoothness or shakiness of the lines. Are they consistent?
                    2.Shape and Structure: Check for similarities in the curves and angles of letters or shapes.
                    3.Spacing: Analyze the space between letters or sections of the signatures.
                    4.Additional Marks: Notice any unique flourishes or additional marks that one has but the other does not.
                    Provide a confidence level on how similar they are based on the above 4 points.
                    """

                    #"Describle in detail the following? Is there any similiarity between them? Provide a confidence level on how similar they are."# Also provide your confidence level in the final result."
                },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url":f"data:image/jpeg;base64,{desc1}" 
                        }
                    },
                     {
                    "type": "image_url",
                    "image_url": {
                        "url":f"data:image/jpeg;base64,{desc2}" 
                        }
                    },                   
                ]
            }
        ],
        model="gpt-4o-mini-prd-gcc2-lb",)

    return response.choices[0].message.content

# Function to convert image to base64
def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

#Function to convert PDF to image for signature checks.
def jpgmaker(pdf_document,filename):

        image_paths = []  # Initialize an empty list to store image paths
    
    # Iterate through each page
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            #st.image(img, caption=f'Page {page_num + 1}', use_column_width=True)
            
            # Save image with the original filename and page number
            img_path = os.path.join('uploads', f'{filename}_page_{page_num + 1}.jpg')
            img.save(img_path, 'JPEG')
            #st.write(f"Saved page {page_num + 1} as {img_path}")

            # Append the image path to the list
            image_paths.append(img_path)

        # returns total pages   
        return len(pdf_document), image_paths


#function to display image locally.
def show_image(img):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


# Function to detect and crop the signature area
def detect_signature_area(np_image):
    # Convert to grayscale
    gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    # Apply binary threshold
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming the largest contour is the signature
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the signature area
    signature_area = np_image[y:y+h, x:x+w]
    return signature_area

# Function to dynamically scales a template image to find and crop the best matching region within a main image, returning this region if a match is found, or None otherwise.
def template_matching(main_image_path, template_image_path, min_match_quality=0.8):
    # Load the main image and template
    image = cv2.imread(main_image_path)
    template = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)

    # Verify images are loaded correctly
    if image is None or template is None:
        raise ValueError("Could not load one or both images")
    

    # Convert the main image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0) # Apply Gaussian Blur to reduce noise

    best_match = None
    best_max_val = -1
    best_roi = None

    # Finer scale granularity: 0.5 to 1.5 in increments of 0.05 
    scales = [i / 100.0 for i in range(50, 151, 5)]

    for scale in scales:
        # Resize the template based on the scale factor
        resized_template = cv2.resize(template, (int(template.shape[1] * scale), int(template.shape[0] * scale)), interpolation=cv2.INTER_AREA)

        # Skip if template is too small
        if resized_template.shape[0] < 10 or resized_template.shape[1] < 10:
            continue

        # Apply template matching
        result = cv2.matchTemplate(gray_image, resized_template, cv2.TM_CCOEFF_NORMED)

        # Get the best match position
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > best_max_val and max_val >= min_match_quality:
            best_max_val = max_val
            best_match = max_loc
            best_template_size = resized_template.shape

            # Get the dimensions of the resized template
            h, w = best_template_size
            # Crop the region of interest (ROI) from the main image
            top_left = best_match
            best_roi = image[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]

    return best_roi

# Secondary Function using signature_extract package
def check_sign(file_path, reference_signature_path):
    loader = Loader()

    try:

        #It returns a list of the masks. Each mask is a numpy 2 dimensions array. Its element's value is 0 or 255.
        mask = loader.get_masks(file_path)[0]

        st.image(mask)

        #Extractor reads a mask, labels the regions in the mask, and removes both small and big regions. We consider that the signature is a region of middle size.
        extractor = Extractor(min_area_size = 5,amplfier=25)

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
            #st.image(result["cropped_mask"])
            if is_signed:
                final_image= result["cropped_mask"]
                break

        if not is_signed:

            st.write("no signature")

        else :
            reference_signature = cv2.imread(reference_signature_path, cv2.IMREAD_GRAYSCALE)
                                
            img_tensor = preprocess_image(final_image)
            ref_img_tensor = preprocess_image(reference_signature)

            # Extract features
            img_features = extract_features(img_tensor)
            ref_img_features = extract_features(ref_img_tensor)

            # Compare features
            similarity_score = cosine_similarity(img_features, ref_img_features)
            
            st.write(f"Signature match score: {similarity_score}")
            st.write("Explanation:")
            st.write("""
            - **Score close to 1**: The two signatures are highly similar, indicating a strong match.
            - **Score close to 0**: The signatures are very different, indicating a weak match.
            - **Typical Thresholds**: In practice, a score above 0.7 might indicate a good match, but this can vary depending on the use case.
            """)

            st.write("This is the identified signature, please verify with the original signature:")
            st.image(final_image) 
            return final_image  
    except Exception as e:
        st.write(f"Error processing file: {e}")


def find_signature(file_path, reference_signature_path):
            #try to detect image
        #np_image = np.array(Image.open(uploaded_file))
        signature_area=template_matching(file_path, reference_signature_path )
        if signature_area is not None:
            
            ####signature_area = detect_signature_area(np_image)
            final_image = preprocess_image(signature_area)
            reference_signature = cv2.imread(reference_signature_path, cv2.IMREAD_GRAYSCALE)
            ref_signature=preprocess_image(reference_signature)

            img_features = extract_features(final_image)
            ref_img_features = extract_features(ref_signature)

            similarity_score = cosine_similarity(img_features, ref_img_features)
            
            st.write(f"Signature match using ResNet score: {similarity_score:.4f} Below is the identified signature")
            st.image(signature_area)
            st.write("Below is the reference signature")
            st.image(reference_signature)
            st.write("Explanation:")
            st.write("""
            - **Score close to 1**: The two signatures are highly similar, indicating a strong match.
            - **Score close to 0**: The signatures are very different, indicating a weak match.
            - **Typical Thresholds**: In practice, a score above 0.7 might indicate a good match, but this can vary depending on the use case.
            """)
            return signature_area

        else:
            return None 
            #st.write("signature not found using template matching, trying signature_detection which is of lower accuracy.Do take note")
            #signature_area=check_sign(file_path, reference_signature_path)
            #if signature_area is not None:
            #    return signature_area
            #return None

##Main 

Total=[]
option=""

with st.expander("Disclaimer"):

 st.write(

"""

IMPORTANT NOTICE: This web application is developed as a proof-of-concept prototype. The information provided here is NOT intended for actual usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.

Always consult with qualified professionals for accurate and personalized advice.

"""

)



st.title("Capstone Draft:Signature Comparison")

#uploaded_file=""

uploaded_file = st.file_uploader("Upload a PDF to validate the signature.",type=['pdf','jpg','png'])
reference_signature_file = st.file_uploader("Upload the reference signature for comparison.", type=['jpg', 'png'])

# Display uploaded image & save the image as the image uploaded is in memory.
if uploaded_file and reference_signature_file:

# Create 'uploads' directory if it doesn't exist
    if not os.path.exists('uploads'):
            temp_dir=os.makedirs('uploads')    

# Extract the original filename without extension
    original_filename = os.path.splitext(uploaded_file.name)[0]
    reference_signature_path = os.path.join('uploads', reference_signature_file.name)
    with open(reference_signature_path, "wb") as f:
        f.write(reference_signature_file.getvalue())
# check if it is PDF format and convert it.   
    if uploaded_file.type == "application/pdf":
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        pages, image_paths=jpgmaker(pdf_document,original_filename)    
        
        for page_num in range(pages):

            #file_path = os.path.join('uploads', f'{original_filename}_page_{page_num + 1}.jpg')
            signature_area=find_signature(image_paths[page_num],reference_signature_path)
            #check_sign(file_path, reference_signature_path)

            Total.append(signature_area)
            

            if signature_area is not None:

                st.write(f"Signature is found on page {page_num+1}")



        # Find indices and values of None elements
        non_none_indices = [(i, x) for i, x in enumerate(Total) if x is not None]     

        
        st.write("Summary")
        selected_pages = []

        for i, x in non_none_indices:
            if st.checkbox(f"Signature are found on page {i+1}" , key=i):
                selected_pages.append(i)

        #Use Case 2 : GPT assistance

        

        st.write("Select an option: Do you want AI to help validate the results?")
        
        # Display a button, which is only enabled if at least one checkbox is selected 
        proceed = st.button("Proceed") if selected_pages else False
    
        end=st.button("Exit. Thank you !")
        if end:
            st.image("goodbye.jpg")
            st.stop()

        if proceed:


                st.write('Summoning AI bot for assistance.....')
                
                for i in selected_pages:
                    # Describe images using GPT-4 Vision
                    reference_signature = cv2.imread(reference_signature_path, cv2.IMREAD_GRAYSCALE)
                    pil_image = Image.fromarray(np.uint8(Total[i]))
                    ref_pil_image = Image.fromarray(np.uint8(reference_signature), mode='L')

                    signature_area_base64 = encode_image_to_base64(pil_image)
                    ref_signature_base64 = encode_image_to_base64(ref_pil_image)

                    #conclusion=find_image_with_gpt4(signature_area_base64,ref_signature_base64)
                    conclusion=decide_gpt4(signature_area_base64,ref_signature_base64)
                    st.write(f"Analysing signature pair at page {i+1}")
                    st.write(f"Conclusion : {conclusion}")

                    #elif option == 'No':
                    #st.write('Send AI bot home..')
                    #st.stop()

        else: st.write("Please select at least one checkbox to proceed.")
        

    else:   




            
            # Define the path where the file will be saved
        file_path = os.path.join('uploads', uploaded_file.name)

            # Save the uploaded file to the temporary directory
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        signature_area=find_signature(file_path,reference_signature_path)


            # Display the file path
        #st.write(f"File saved at: " + file_path)

        # Old method of using signature_extractor
        #check_sign(file_path, reference_signature_path)

        option = st.selectbox(
            "Select an option: Do you want AI to help validate the results?",
            ('Select an option...','Yes', 'No')
        )

        if option == 'Yes':
            st.write('Summoning AI bot for assistance.....')
            
            # Describe images using GPT-4 Vision
            reference_signature = cv2.imread(reference_signature_path, cv2.IMREAD_GRAYSCALE)
            pil_image = Image.fromarray(np.uint8(signature_area))
            ref_pil_image = Image.fromarray(np.uint8(reference_signature), mode='L')

            signature_area_base64 = encode_image_to_base64(pil_image)
            ref_signature_base64 = encode_image_to_base64(ref_pil_image)

            conclusion=decide_gpt4(signature_area_base64,ref_signature_base64)
            st.write(f"Conclusion : {conclusion}")

        elif option == 'No':
            st.write('Goodbye then.')
            st.stop()



        


            
        #signature_area=check_sign(file_path, reference_signature_path)

        # Describe images using GPT-4 Vision
            #pil_image = Image.fromarray(np.uint8(signature_area))
            #ref_pil_image = Image.fromarray(np.uint8(reference_signature), mode='L')

            #signature_area_base64 = encode_image_to_base64(pil_image)
            #ref_signature_base64 = encode_image_to_base64(ref_pil_image)


        


        #description_1 = describe_image_with_gpt4(signature_area_base64)
        #description_2 = describe_image_with_gpt4(ref_signature_base64)
       
        #conclusion = decide_gpt4(signature_area_base64,ref_signature_base64)

        #st.write("GPT-4 Vision Descriptions:")
        #st.write(f"Description 1: {description_1}")
        #st.write(f"Description 2: {description_2}")
        #st.write(f"Final conclusion : {conclusion}")



    
