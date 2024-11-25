import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests  # Added for downloading model

def download_model(url):
    response = requests.get(url, allow_redirects=True)
    if response.status_code == 200:
        with open("plant_disease_model.keras", "wb") as f:
            f.write(response.content)
        return "Model downloaded successfully!"
    else:
        return f"Error downloading model: {response.status_code}"

# Download model from Google Drive (replace with your actual link)
model_download_message = download_model("https://drive.google.com/uc?export=download&id=1-3ruHJ25v1o3_UWQOGFOGArYuxeg6wye")
st.info(model_download_message)  # Display download status

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('plant_disease_model.keras')
    return model

def preprocess_image(image):
    img = image.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.title('Plant Disease Classification (Potato)')

    # Display the provided default potato image
    default_image_url = "https://img.freepik.com/premium-psd/falling-fresh-potatoes-transparent-background_220739-468.jpg"
    st.image(default_image_url, caption='Example Potato Image', use_column_width=True)

    uploaded_file = st.file_uploader("Choose a potato leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded 1  Image', use_column_width=True)

        model = load_model()
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
