import streamlit as st
import pandas as pd

# Model data
data = {
    'Weight': ['ResNet50_Weights.IMAGENET1K_V2'],
    'Acc@1': [80.858],
    'Acc@5': [95.434],
    'Params': ['25.6M'],
    'GFLOPS': [4.09]
}

# Create a DataFrame
df = pd.DataFrame(data)


# Setting the title
st.title("About Us")

# Project Scope
st.header("Project Scope")
st.write(
    """
    In WOG Ideathon, our users highlighted a significant challenge they face monthly. They receive an extensive volume of documents and job reports, which require thorough examination to ensure that every single page contains an authorized signature. 
    This manual verification process is not only time-consuming but also prone to human error, leading to inefficiencies and reduced productivity. 
    The users expressed a strong desire for an automated solution that could handle this repetitive task, allowing them to focus on more critical aspects of their jobs and significantly enhancing their overall productivity.
    """
    )

# Objectives
st.header("Objectives")
st.write(
    """
    The primary objective of this project is to develop an automated solution that verifies the presence of authorized signatures on every page of received documents and job reports.
    This solution aims to reduce the manual effort involved, minimize errors, and improve productivity for our users.
    
    """
    )

# Data Sources
st.header("Data Sources")
st.write(
        """

    As this is an image detection , we are using PyTorch ResNet 50 pre-trained model for the image detection. 
    It has one of the highest accuracy reported and is widely used for image detection. 
    We tried experimenting with the other weight but the results are not as good as ResNet50 model so finally we decided with ResNet50

    Accuracies are reported on ImageNet-1K using single crops:
        
        """
        )
st.table(df)


# Features
st.header("Features")
st.subheader("Use Case 1: Signature Identification")
st.write(
    """
        
    To enhance productivity and accuracy in document verification, our solution employs a image matching algorithm tailored to detect authorized signatures across multiple pages within documents or reports.

    1.Signature Extraction:
    
    The process begins by extracting the authorized signature from a reference document. This signature serves as the benchmark for all subsequent image matching operations.
    (For the Proof of Concept (POC), we'll permit the upload of authorized signatures to test and validate the system's functionality. However, in a real-world application, strict security measures will ensure that authorized signatures are not uploaded or exposed, maintaining compliance and safeguarding sensitive information.)
    
    2.Image Matching Algorithm:
    
    The core functionality involves using advanced image matching algorithms. These algorithms compare the reference signature against each page of the document or report.
    
    To ensure thoroughness, the algorithm applies various image processing techniques such as rotation, scaling, and translation. This allows it to accommodate variations in signature size, orientation, and positioning across different pages.
    
    3.Scaling for Best Match:
    
    The algorithm dynamically scales the reference signature to different sizes. This scaling ensures that even if the signature appears larger or smaller on certain pages, the best match can be found.
    
    By iterating through multiple scaling factors, the algorithm increases the likelihood of accurately identifying the authorized signature regardless of its size variations.
    
    4.Scoring :
    
    Once the image matching process is complete, the algorithm assigns a confidence score to each detected signature. This score reflects the likelihood that the identified signature matches the reference.
    
    """
    )

st.subheader("Use Case 2: Leveraging OpenAI's Vision Capabilities for Secondary Verification")
st.write(
    """
         
    To enhance the accuracy and reliability of our signature verification system, we utilise OpenAI's new vision capabilities as a secondary layer of verification. 
    By employing this advanced technology, we aim to ensure that the identified signature within the document is authenticated with the highest confidence. 
    The process involves the following key steps:

    For each identified signature, we harness OpenAI's latest vision capabilities for a secondary level of verification. This step utilizes cutting-edge technology to analyze the signatures in greater detail.

    Guiding Criteria for Comparison:
    
        To guide the OpenAI vision model in performing accurate comparisons, we establish four main criteria:
        
        Shape Similarity: Analyzes the geometric shape and contours of the signature.
        
        Position Accuracy: Evaluates the positional alignment of the signature within the document.
        
        Signature Size: Considers variations in signature size and scales accordingly.
        
        Visual Characteristics: Examines finer details such as stroke thickness, angles, and unique identifying marks.

    Confidence Level Assessment:

    Based on the comparison of these criteria, OpenAI's vision model provides a confidence level for each assessed signature. 
    This confidence score indicates the likelihood that the identified signature matches the authorized signature.
    """
    )


