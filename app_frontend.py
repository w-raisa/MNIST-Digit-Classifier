# app_frontend.py

# external dependencies
import tensorflow as tf 
from tensorflow.keras.models import load_model # imprting func we need to load model

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from skimage.transform import resize # to resize the submitted canvas img to 28 x 28 as our mdoel accepts 28 x 28 imgs
from skimage.color import rgb2gray, rgba2rgb # submitted canvas is rgb (has 3 channels) need to convert it to gray scale because our model accepts only gray scale imgs

from scipy.special import softmax


# puts whatever is being returned by the function in a cache (so here its a tf model)
def load_pretrained_model(file_path = "CNN_model"):
    return load_model(file_path) # returns model in file_path

model = load_pretrained_model()

def normalize_img(image: np.ndarray):
    """Normalizes images: `uint8` -> `float32`."""
    return 1. - image.astype(np.float32) / 255.


# Specify canvas parameters in application
# NOTE: the st.sidebar prefix simply indicates that the streamlit object
#       will be placed in the sidebar rather than in the web app's main body
stroke_width = st.sidebar.slider("Stroke width: ", min_value=20, max_value=30, step=1, value=20)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
drawing_mode = "freedraw"

col1, col2, col3 = st.columns([1,6,1])

with col2:
    st.title("Digit Detector")

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color = "rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width = stroke_width,
        stroke_color = stroke_color,
        background_color = bg_color,
        # keep this as False to avoid re-running the app whenever the user starts drawing
        update_streamlit=True, 
        height=224,
        width=224, #in pixels 
        drawing_mode=drawing_mode,
        key="canvas",
    )


    if st.button("SUBMIT"):
        # stuff below only gets run once the user hits the "SUBMIT" button
        grayscale_img = normalize_img(
            np.sum(canvas_result.image_data[:,:,:3], axis=-1)
        ) # turn img grayscale
        resized_image = resize(grayscale_img, (28,28))
        
        signals = model.predict(np.expand_dims(resized_image, axis=0)).flatten()

        # change signals into probabilities using softmax
        probabilities = softmax(signals)
        values = ['0', '1', '2', '3','4','5', '6', '7', '8', '9']
        plt.style.use('seaborn')
        plt.rcParams.update({'text.color': "white",
                        'axes.labelcolor': "white"})
        fig, ax = plt.subplots()
        ax.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
        ax.tick_params(axis='y', colors='white')  #setting up Y-axis tick color to black
        ax.bar(list(range(10)), probabilities)
        plt.xticks(list(range(0,10)), values)
        plt.title("Predictions")

        ax.set_xlabel('Digit', fontweight ='bold', )
        ax.set_ylabel('Probabilities', fontweight ='bold')
        st.pyplot(fig, transparent=True)



