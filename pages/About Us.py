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
st.write(
    """
    #Use Case 1: Signature Identification
    
    To enhance productivity and accuracy in document verification, our solution employs a sophisticated image matching algorithm tailored to detect authorized signatures across multiple pages within documents or reports.

    **Signature Extraction:
    
    The process begins by extracting the authorized signature from a reference document. This signature serves as the benchmark for all subsequent image matching operations.
    
    **Image Matching Algorithm:
    
    The core functionality involves using advanced image matching algorithms. These algorithms compare the reference signature against each page of the document or report.
    
    To ensure thoroughness, the algorithm applies various image processing techniques such as rotation, scaling, and translation. This allows it to accommodate variations in signature size, orientation, and positioning across different pages.
    
    **Scaling for Best Match:
    
    The algorithm dynamically scales the reference signature to different sizes. This scaling ensures that even if the signature appears larger or smaller on certain pages, the best match can be found.
    
    By iterating through multiple scaling factors, the algorithm increases the likelihood of accurately identifying the authorized signature regardless of its size variations.
    
    **Scoring Mechanism:
    
    Once the image matching process is complete, the algorithm assigns a confidence score to each detected signature. This score reflects the likelihood that the identified signature matches the reference.
    
    The scoring mechanism considers various factors, including shape similarity, position accuracy, and visual characteristics.


    We leverage on OpenAI's new vision capabilities to help provide a secondary level of verification from the identified signature in the document. 
    We provide 4 main critiera to help guide the OpenAI to do the comparison and finally to provide a confidence level on its assessments.


    
    """
    )


