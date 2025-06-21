import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

working_dir = os.path.dirname(os.path.abspath(__file__))
print(working_dir)
model_path = f'{working_dir}/trained_models/plant_disease_prediction.h5'

#Load the Model
model = tf.keras.models.load_model(model_path)

#class indices
class_indices = json.load(open(f'{working_dir}/class_indices.json'))

# Functions

def load_and_preprocess_img(image_path, target_size=(224,224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array,axis=0)
    img_array = img_array.astype('float32')/255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed = load_and_preprocess_img(image_path)
    prediction = model.predict(preprocessed)
    class_index = np.argmax(prediction, axis=1)[0]
    class_name = class_indices[str(class_index)]
    return class_name

# StreamLit App

st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an Image...", type=['jpg','jpeg','png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1,col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150,150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            prediction = predict_image_class(model,uploaded_image,class_indices)
            st.success(f'Prediction: {str(prediction)}')