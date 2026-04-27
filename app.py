import streamlit as st
import pandas as pd
import joblib

model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')


# @title
st.title('Heart Stroke Prediction')
st.markdown("Provide the following details")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("SEX", ['M', 'F'])
chest_pain = st.selectbox("Chest Pain Type", ['ATA','NAP','TA','ASY'])
Resting_BP =st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
Cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
Fasting_bs =  st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
Resting_ECG = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ['y', 'n'])
oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 6.0 , 1.0)
st.slope = st.selectbox("Slope of the Peak Exercise ST Segment", ['Up', 'Flat', 'Down'])


if st.button("Predict"):
    raw_input = {
        'age': age,
        'sex' + sex: 1,
        'chest_pain'+ chest_pain : 1,
        'resting_bp': Resting_BP,
        'cholesterol': Cholesterol,
        'fasting_bs': Fasting_bs,
        'resting_ecg'+ Resting_ECG : 1,
        'max_hr': max_hr,
        'exercise_angina' + exercise_angina: 1,
        'oldpeak': oldpeak,
        'slope' + st.slope: 1
    }
    input_data = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0 
    input_data = input_data[expected_columns]  

    input_data_scaled = scaler.transform(input_data)   

    prediction = model.predict(input_data_scaled)[0]   

    if prediction == 1:     
        st.error("High risk of heart stroke. Please consult a doctor.") 
    else:
        st.success("Low risk of heart stroke. Keep up the healthy lifestyle!")  
