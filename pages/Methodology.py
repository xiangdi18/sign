import streamlit as st

# Add custom HTML and CSS to style the caption 
st.markdown( """ <style> .caption { font-size: 20px; font-family: Arial, sans-serif; color: blue; } </style> """, unsafe_allow_html=True) 




st.title("Methodology (Flow Chart)")
st.image("method.jpg")
st.markdown('<div class="caption">Signature verfication flow</div>', unsafe_allow_html=True)

st.write(
"""
#Use Case 1 : Signature verfication flow
Step 1:

User requires to upload 2 document, the actual document for verification and 2nd the authorised signature for verification.

User uploads pdf or jpg document which contains the signature for verification, streamlit built-in checks to ensure only pdf or jpg document is allowed to upload. 

For signature, we only allow jpg to ensure that only 1 signature is uploaded. 

As streamlit stores documents inside memory. we would need to convert into jpg images first before we can perform any image manipulation.

jpgmaker() is used to convert any pdf document to jpg and stores into upload folder.

    a)It extracts the filename and appends page number to it.
    b)Finally, it returns the total pages , array of jpg location.



For the sake of this POC, we allow user to upload signature for verification. 

***For actual production use case, the authorised signature will be uploaded/saved privately and users only need to upload the document for check.


Step 2:

We create a find_signature() function that helps to identify the signature to the authorised signature.

Looping through each page of the document to the find_signature() to help to identify the signature area and returns the results to user at the end.

find_signature() consists of a few sub modules to help with the signature detection.

First: we call the template_matching() to try to find the identified area of signature in the document

    template_matching() - Function to dynamically scales the authorised image(signature) to find and crop the best matching region within a main image, returning this region if a match is found, or None otherwise.
        a) converts the uploaded image to grayscale and apply Gaussian Blur to reduce noise from the image.
        b) resize the authorised signature using scale granularity: 0.5 to 1.5 in increments of 0.05  (tried using less granular but results are not as good hence make it smaller and more granular however will have performance impact)
        c) apply cv2.matchTemplate to try to find the signature using cv2.TM_CCOEFF_NORMED method.
        d) Depending on the return results High values indicate better matches (for methods like cv2.TM_CCOEFF_NORMED) 
        e) We keep the highest value (best match) image and crop the region of interest (ROI) from the main image using the best match coordinates
        f) Return the cropped region if a match is found, otherwise return None

Second: Once we got the area in numpy array format, we have to convert it preprocess_image() to an PyTorch format to do forward processing.

    preprocess_image() - The preprocess_image function takes a NumPy image array and preprocesses it for features extraction.

Third: After preparing the image send to a pre-trained ResNet-50 model to extract the features from the image.

    extract_features() - Pass the input image through the model to extract features with features = model(image) and return the flattened feature vector to use for comparison.

Fourth: We use cosine_similarity() to compare the 2 features , basically it measures the cosine of the angle between them and returns a scoring.

    cosine_similarity() - Use torch.nn.functional.cosine_similarity to compute the cosine similarity between the two feature vector and return the cosine similarity score.

Fifth:

    Return the result of the image with the scoring.

Step 3:

Displaying the idetified signature with the authorised signature on the found page together with scoring for user's verification.

Provides a summary on the signature found on pages xxx out of the whole document.

The End.

# Use Case 2 : AI assisted verification 

For the use case 2 where we try to muster chatGPT's vision to help to verify the identified signature with the authorised signature and to provide an confidence level to the user.

Step 4:

We encode the image back to base64 as OpenAI 's vision capabilities only works with that.

then we use decide_gpt4() function to ask the OpenAI's vision to compare the 2 images based on below:

    1.Line Quality: Look at the smoothness or shakiness of the lines. Are they consistent?
    2.Shape and Structure: Check for similarities in the curves and angles of letters or shapes.
    3.Spacing: Analyze the space between letters or sections of the signatures.
    4.Additional Marks: Notice any unique flourishes or additional marks that one has but the other does not.
                    
Finally, provide a confidence level on the check to the user.

The end of use case 2.

"""
)




