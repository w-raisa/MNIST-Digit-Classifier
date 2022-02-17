# app_frontend.py

# external dependencies
#import tensorflow as tf 
#from tensorflow.keras.models import load_model

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# high-level overview #
# 1. allow the user to draw a digit/something on a canvas
# 2. allow the user to submit the image for prediction/inference
# 3. image -> resizing image -> model -> 10x1 numpy array of probabilities ->
#    ... -> transfer information to the front-end -> display probabilities as a histogram

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
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    # keep this as False to avoid re-running the app whenever the user starts drawing
    update_streamlit=False, 
    height=150,
    drawing_mode=drawing_mode,
    key="canvas",
)
if st.button("SUBMIT"):
    # stuff below only gets run once the user hits the "SUBMIT" button
    probabilities = predict()
    st.markdown(str(probabilities))
    fig, ax = plt.subplots()
    ax.hist(probabilities)
    st.pyplot(fig)
