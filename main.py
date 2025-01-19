import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# Keras Model Prediction
def model_prediction(test_image):
    # Load the model using keras
    model = load_model("vgg16_model2.h5")
    
    # Preprocess the image
    image = load_img(test_image, target_size=(256, 256))
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    input_arr /= 255.0  # Normalize the image to [0, 1]
    
    # Predict using the model
    predictions = model.predict(input_arr)
    
    # Return the index of the maximum element in predictions
    return np.argmax(predictions)

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","Disease Classification"])

#Main Page
if(app_mode=="Home"):
    st.header("RICE PLANT DISEASE CLASSIFICATION SYSTEM")
    image_path = "riceplant.png"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Rice Plant Disease Classification System! 🌿🔍
    
    The mission is to help in identifying rice plant diseases efficiently. Upload an image then the system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Classification** page and upload an image of a rice plant with suspected diseases.
    2. **Analysis:** The system will process the image using VGG16 model to identify potential diseases.
    3. **Results:** View the results of the disease.

    ### About
    - **Accuracy:** The system utilizes transfer learning techniques for accurate disease classification.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Classification** page in the sidebar to upload an image and experience for Rice Plant Disease Classification System!

    #### About Dataset
                
    """)

    image_path = "disease.png"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    The trained dataset can be found on the link [Kaggle Rice Crop Diseases Dataset](https://www.kaggle.com/datasets/thegoanpanda/rice-crop-diseases). 
    This dataset contains 200 images of disease-infected rice plants. The images are grouped into 4 classes based on the type of disease. There are 50 images in each class.
    - Bacterial Leaf Blight Disease
    - Blast Disease
    - Brown Spot Disease
    - False Smut
    """)

#Prediction Page
elif(app_mode=="Disease Classification"):
    st.header("Disease Classification")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    if test_image is not None:
        # Display the uploaded image
        st.image(test_image, width=200)
    #Predict button
    if(st.button("Predict")):
        st.write("Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Bacterial Blight Disease', 'Blast Disease', 'Brown Spot Disease', 'False Smut Disease']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))

        st.header("Example and Characteristics")
        if(class_name[result_index]=='Bacterial Blight Disease'):
            image_path = "blb.png"
            st.image(image_path, width=200,caption='example')
            st.markdown("""
            Characteristics:
            - Caused by Xanthomonas Oryzae Pvoryzae.
            - Water-soaked spots appear at the edges of leaves.
            - Infected areas expand in length and width.
            - Color changes from yellow to light brown due to drying.
            """)
        elif(class_name[result_index]=='Blast Disease'):
            image_path = "blast.png"
            st.image(image_path, width=200,caption='example')
            st.markdown("""
            Characteristics:
            - Caused by Magnaporthe oryzae.
            - Circular spots with grayish-green centers surrounded by dark green borders appear on leaves.
            - Spots elongate and develop reddish-brown margins as the disease progresses.
            - Eventually causes total damage to the leaves.
            """) 
        elif(class_name[result_index]=='Brown Spot Disease'):
            image_path = "bs.png"
            st.image(image_path, width=200,caption='example')
            st.markdown("""
            Characteristics:
            - Caused by the fungus Bipolaris oryzae.
            - Dark brown to reddish-brown small spots appear on parts of the plant above the ground, especially leaves.
            - Common in all rice-growing regions, particularly under conditions of poor fertilization and water scarcity.
            - Leads to stunted rice growth.
            """) 
        elif(class_name[result_index]=='False Smut Disease'):
            image_path = "fs.png"
            st.image(image_path, width=200,caption='example')
            st.markdown("""
            Characteristics:
            - Caused by Ustiloginoidea virens.
            - Symptoms appear on rice panicle grains.
            - Spherical spore balls, 1 cm or larger, cover the florets.
            - Color transitions from yellow to orange, then to yellowish-green.
            """)             