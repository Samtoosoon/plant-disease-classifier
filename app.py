import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

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
    st.title('Plant Disease Classification')
    
    uploaded_file = st.file_uploader("Choose a plant leaf image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
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