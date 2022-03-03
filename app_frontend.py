# app_frontend.py

# external dependencies
import tensorflow as tf 
from tensorflow.keras.models import load_model # imprting func we need to load model

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from skimage.transform import resize # to resize the submitted canvas img to 28 x 28 as our mdoel accepts 28 x 28 imgs
from skimage.color import rgb2gray # submitted canvas is rgb (has 3 channels) need to convert it to gray scale because our model accepts only gray scale imgs

# high-level overview #
# 1. allow the user to draw a digit/something on a canvas
# 2. allow the user to submit the image for prediction/inference
# 3. image -> resizing image -> model -> 10x1 numpy array of probabilities ->
#    ... -> transfer information to the front-end -> display probabilities as a histogram

# puts whatever is being returned by the function in a cache (so here its a tf model)
#@st.cache(suppress_st_warning=True) # this is a decorator, tells streamlit load this once and only once (cuz streamlit runs from top to bottom everytime soemthing happens on website). 
def load_pretrained_model(file_path = "CNN_model"):
    return load_model(file_path) # returns model in file_path


model = load_pretrained_model()


def predict():
    # placeholder for the function that will handle 
    # model predictions
    # input:    nothing for now, will be an image
    # output:   numpy vector of probabilities such that 
    #           vector[0] == probability that the image is a 0
    vector = np.random.rand(10)
    return vector

st.title("LIVE MNIST")

st.markdown("# THIS IS A TITLE")

# Specify canvas parameters in application
# NOTE: the st.sidebar prefix simply indicates that the streamlit object
#       will be placed in the sidebar rather than in the web app's main body
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
# bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
#drawing_mode = st.sidebar.selectbox(
#    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
#)
drawing_mode = "freedraw"

# Create a canvas component
canvas_result = st_canvas(
    fill_color = "rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width = stroke_width,
    stroke_color = stroke_color,
    background_color = bg_color,
    # keep this as False to avoid re-running the app whenever the user starts drawing
    update_streamlit=False, 
    height=224,
    width=224, #in pixels 
    drawing_mode=drawing_mode,
    key="canvas",
)

if st.button("SUBMIT"):
    # stuff below only gets run once the user hits the "SUBMIT" button
    grayscale_img = rgb2gray(canvas_result.image_data) # turn img grayscale
    resized_image = resize(grayscale_img, (28,28))

    probabilities = model.predict(np.expand_dims(resized_image, axis=0))
    st.markdown(str(probabilities))
    fig, ax = plt.subplots()
    ax.scatter(list(range(10)), probabilities)
    st.pyplot(fig)

    # resize img to 28 x 28 img
    #st.image(canvas_result.image_data)

    #st.image(grayscale_img)
    #print(resized_image)


    # pass resized img to model

