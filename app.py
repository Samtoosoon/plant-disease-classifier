import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os

# Function to download the model from Google Drive using a direct download link
def download_model(url, model_name='plant_disease_model.keras'):
    # Modify the URL to convert it into a direct download link
    file_id = url.split('/d/')[1].split('/')[0]
    download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
    
    response = requests.get(download_url, allow_redirects=True)
    if response.status_code == 200:
        with open(model_name, "wb") as f:
            f.write(response.content)
        return f"Model downloaded successfully as {model_name}!"
    else:
        return f"Error downloading model: {response.status_code}"

# Download model from Google Drive (replace with your actual link)
model_url = "https://drive.google.com/file/d/1-6W-u-KoyL511t-OVuCQY--YZL93NB6N/view?usp=sharing"
model_download_message = download_model(model_url)
st.info(model_download_message)  # Display download status

# Load the downloaded model from a local file
@st.cache_resource
def load_model(model_path='plant_disease_model.keras'):
    # Check if the model exists
    if not os.path.exists(model_path):
        st.error("Model file not found. Please download the model first.")
        return None
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(image):
    img = image.resize((256, 256))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def main():
    st.title('Plant Disease Classification (Potato)')

    # Display the provided default potato leaf image
    default_image_url = "https://png.pngtree.com/png-vector/20230912/ourmid/pngtree-illustration-of-a-simple-leaf-png-image_10027828.png"
    st.image(default_image_url, caption='Example Leaf Image', use_column_width=True)

    uploaded_file = st.file_uploader("Choose a potato leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        model = load_model()  # Load the model

        if model:
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction)
            confidence = round(100 * np.max(prediction), 2)

            class_names = ['Healthy', 'Lateblight', 'Earlyblight']

            st.subheader('Prediction Results')
            st.write(f'Predicted Class: {class_names[predicted_class]}')
            st.write(f'Confidence: {confidence}%')

if __name__ == '__main__':
    main()
