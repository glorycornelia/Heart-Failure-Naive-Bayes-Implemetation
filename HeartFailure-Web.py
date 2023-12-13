import pickle
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu


# loading the saved models
heart_failure_model = pickle.load(open('gnb_model.sav','rb'))
oe = pickle.load(open('ordinal_encoder.sav', 'rb')) 
scaler = pickle.load(open('minMax_scaler.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Prediction System',
                          
                          ['Heart Failure Prediction'],
                          icons=['heart'],
                          default_index=0)
    

# Heart Failure Prediction Page
if (selected == 'Heart Failure Prediction'):
    
    # page title
    st.title('Heart Failure Prediction using Naive Bayes')
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col1:
        cp = st.text_input('Chest Pain types')
        
    with col2:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col1:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col2:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col1:
        exang = st.text_input('Exercise Induced Angina')
        
    with col2:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col1:
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
        print(input_data)

        input_data=scaler.transform(input_data)
        print(input_data)

        heart_prediction = heart_failure_model.predict(input_data)                          

        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart Failure'
        else:
          heart_diagnosis = 'The person does not have any heart Failure'
        
    st.success(heart_diagnosis)