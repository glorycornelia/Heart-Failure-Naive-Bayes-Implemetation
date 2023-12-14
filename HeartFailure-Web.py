import json
import pickle
import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

st.set_page_config(layout="wide")

# Load lottie file
loaded_lottie = load_lottiefile('Animation/heart_animation.json')

# loading the saved models
heart_failure_model = pickle.load(open('gnb_model.sav','rb'))
oe = pickle.load(open('ordinal_encoder.sav', 'rb')) 
scaler = pickle.load(open('minMax_scaler.sav', 'rb'))
    

# Heart Failure Prediction Page
col1, col2 = st.columns(2)

with col1:
    # page title
    st.title('Heart Failure Prediction using Naive Bayes')
    st.write('Web aplication that implements Naive Bayes method to predict Heart Failure Desease. Please fill the form on the right and start your prediction!')
    st.markdown("---")
    st_lottie(loaded_lottie, speed=0.5,reverse=False,loop=True,quality="low",height=500,width=500)

with col2:
    age = st.text_input('Age')
    sex = st.text_input('Sex')
    cp = st.text_input('Chest Pain types')
    trestbps = st.text_input('Resting Blood Pressure')
    chol = st.text_input('Serum Cholestoral in mg/dl')
    fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    restecg = st.text_input('Resting Electrocardiographic results')
    thalach = st.text_input('Maximum Heart Rate achieved')
    exang = st.text_input('Exercise Induced Angina')
    oldpeak = st.text_input('ST depression induced by exercise')
    slope = st.text_input('Slope of the peak exercise ST segment')
    
    
# code for Prediction
heart_diagnosis = ''

# creating a button for Prediction

if st.button('Heart Failure Test Result'):
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]],
                                columns=['Age', 'Sex', 'ChestPainType',
                                        'RestingBP', 'Cholesterol', 'FastingBS',
                                        'RestingECG', 'MaxHR', 'ExerciseAngina',
                                        'Oldpeak', 'ST_Slope'])

    input_data[['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']]=oe.transform(input_data[['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']])

    input_data=scaler.transform(input_data)

    heart_prediction = heart_failure_model.predict(input_data)                          

    if (heart_prediction[0] == 1):
        heart_diagnosis = 'The person is having heart Failure'
    else:
        heart_diagnosis = 'The person does not have any heart Failure'
    
st.success(heart_diagnosis)

st.markdown("---")

# Add your footer content using Markdown and HTML
footer = """
    <div style="text-align: center;">
        <p>&copy; 2023 Heart Failure Prediction. All rights reserved.</p>
        <p>Designed with ❤️ by Glory, Chienta, Nabila, Lazia</p>
    </div>
"""

st.markdown(footer, unsafe_allow_html=True)