import numpy as np
from tensorflow.keras.models import model_from_json
import streamlit as st

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
loaded_model.compile(loss='binary_crossentropy',
                     optimizer='rmsprop', metrics=['accuracy'])

page_bg_img = '''
<style>
body {
background-image: url("bg_image.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

st.header("Your UNiversity Chance Predictor")

input_marks_GRE = st.number_input(
    label="GRE Score", min_value=260, max_value=340, step=1)
input_marks_TOEFL = st.number_input(
    label="TOEFL Score", min_value=0, max_value=120, step=1)
input_Uni_Rating = st.slider(
    label="Rating", min_value=0, max_value=5, step=1)
input_SOP = st.slider(
    label="SOP", min_value=0.0, max_value=5.0, step=0.5)
input_LOR = st.slider(
    label="LOR", min_value=0.0, max_value=5.0, step=0.5)
input_CGPA = st.slider(
    label="SOP", min_value=0.00, max_value=10.00, step=0.01)
research = -1
st.write("Have you done any research work?")
if st.checkbox("Yes"):
    research = 1
elif st.checkbox("NO"):
    research = 0
if st.button("Rate the Chances"):
    inputs = np.expand_dims([input_marks_GRE, input_marks_TOEFL,
                            input_Uni_Rating, input_SOP, input_LOR, input_CGPA], 0)
prediction = loaded_model.predict(inputs)
print(f"Your Chances are {np.squeeze(prediction, -1):.2f}")
