# streamlit app
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder


#loading the trained model 
model=tf.keras.models.load_model('model.h5')

#load the encoders and scalers 
with open ('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open ('scaler.pkl','rb')as file:
    scaler=pickle.load(file)    


##streamlit app
st.set_page_config(page_title="Churn predictor",layout="wide")
st.title('Customer Churn Prediction')  
st.markdown("Enter the customer details below to predict the likelihood of them leaving the bank") 

#user input
geography =st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score= st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure =st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member= st.selectbox('Is Active Member', [0, 1])

if st.button('Predict Churn'):
## prepare the input data 
  input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
  })

  #ohe "geography"
  geo_encoded=onehot_encoder_geo.transform([[geography]]).toarray()
  geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
  
  #combing ohencoded columns with input data
  input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
  
  
  # scale the input data
  input_data_scaled =scaler.transform(input_data)
  
  #predict churn 
  prediction=model.predict(input_data_scaled)
  prediction_proba=prediction[0][0]
  
  st.markdown(f"<h2 style='color:blue;'>Churn Probability: {prediction_proba:.2f}</h2>", unsafe_allow_html=True)
  
  if prediction_proba>0.5:
        st.markdown("<h3 style='color:red; font-weight:bold;'>The customer is likely to churn.</h3>", unsafe_allow_html=True)
  else :
        st.markdown("<h3 style='color:green; font-weight:bold;'>The customer is not likely to churn.</h3>", unsafe_allow_html=True)
  # to download input data as csv
  csv = input_data.to_csv(index=False)
  st.download_button(
        label="Download Input Data as CSV",
        data=csv,
        file_name='input_data.csv',
        mime='text/csv'
   )