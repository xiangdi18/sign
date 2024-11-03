import streamlit as st

# Setting the title
st.title("About Us")

# Project Scope
st.header("Project Scope")
st.write("This section will provide an overview of the project's scope.")

# Objectives
st.header("Objectives")
st.write("This section will outline the main objectives of the project.")

# Data Sources
st.header("Data Sources")
st.write("""

        As this is an image detection , we are using PyTorch ResNet 50 pre-trained model for the image detection. 
        It has one of the highest accuracy reported and is widely used for image detection. 
        We tried experimenting with the other weight but the results are not as good as ResNet50 model so finally we decided with ResNet50

        Accuracies are reported on ImageNet-1K using single crops:
        
        Weight                            Acc@1   Acc@5   Params GFLOPS
        ResNet50_Weights.IMAGENET1K_V2    80.858  95.434  25.6M  4.09

        
        """
        )

# Features
st.header("Features")
st.write("This section will describe the main features of the project.")


