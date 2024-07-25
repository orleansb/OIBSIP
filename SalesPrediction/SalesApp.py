import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np
import base64

def get_base64_of_image(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

image_path = 'plain.png'
base64_string = get_base64_of_image(image_path)

background_css = f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{base64_string}");
    background-size: cover;
    background-repeat: no-repeat;
    font-colour: blue;
}}

.custom-text{{
    color: black;
}}

</style>
"""

# use CSS with Markdown
st.markdown(background_css, unsafe_allow_html=True)

model = pickle.load(open('rf_model.pkl', 'rb'))
scaler = pickle.load(open("scaler.pkl","rb"))


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data"])

st.title('Sales Prediction App')

if page == "Home":
    st.write("This app allows you to determine your sales depending on the amount spent on different forms of advertising.")
    st.write("Use the sidebar to navigate to different sections of the app.")

# Data Page
elif page == "Data":

    def user_input_features():
        TV = st.number_input(label="Amount Spent on TV Advertisemnt",min_value=0.0, format = "%.2f")
        Radio = st.number_input(label="Amount Spent on Radio Advertisemnt",min_value=0.0,  format = "%.2f")
        Newspaper = st.number_input(label="Amount Spent on Newspaper Advertisemnt",min_value=0.0,  format = "%.2f")

        data = {'TV': TV, 'Radio': Radio,'Newspaper': Newspaper }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    st.session_state['input_df'] = input_df

    st.subheader('Advertisement Amounts')
    st.write(input_df)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    st.subheader('Total Sales:')
    st.write(f'Sales: {prediction[0]:.2f}')


   