import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image

# Model yükleme
model = keras.models.load_model("rock_paper_scissors_model.h5")

class_names = ['paper', 'rock', 'scissors']

st.title("Rock Paper Scissors Classifier")
st.write("Bir görüntü yükleyin, model tahmin yapsın!")

uploaded_file = st.file_uploader("Görüntü seç", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((150,150))
    st.image(img, caption="Yüklenen görüntü", use_container_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = class_names[np.argmax(prediction)]

    st.success(f"Tahmin: {result}")
