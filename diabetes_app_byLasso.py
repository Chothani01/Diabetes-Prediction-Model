import pickle
import pandas as pd
import numpy as np
import streamlit as st
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


uri = "mongodb+srv://princechothani53:HHplNTJ9feCPD2005@cluster0.7bqp1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

db = client["People"]
collection = db["Diabetes"]
# Send a ping to confirm a successful connection


def load_model():
    with open("diabetesbyLasso.pkl", "rb") as f:
        model , scaler = pickle.load(f)
        return model, scaler
    
def process_data(data, scaler):
    df = pd.DataFrame([data])
    processed_data = scaler.transform(df)
    return processed_data

def predict_data(data):
    model, scaler = load_model()
    processed_data = process_data(data, scaler)
    result = model.predict(processed_data)
    return result
    
def main():
    # age: Age in years
    # sex: Gender of the patient
    # bmi: Body mass index
    # bp: Average blood pressure
    # s1: Total serum cholesterol (tc)
    # s2: Low-density lipoproteins (ldl)
    # s3: High-density lipoproteins (hdl)
    # s4: Total cholesterol / HDL (tch)
    # s5: Possibly log of serum triglycerides level (ltg)
    # s6: Blood sugar level (glu)
    
    st.title("Diabetes Analyzer")
    st.title("Enter your details for get your diabetes result prediction")
    
    age = st.number_input("Age", min_value=1, max_value=200, value=5)
    sex = st.radio("Sex", ["Male", "Female"])
    bmi = st.number_input("Bmi", min_value=1.0, max_value=10000.0, value=50.0)
    bp = st.number_input("Average blood pressure", min_value=30.0, max_value=500.0, value=90.0)
    s1 = st.number_input("Total serum cholesterol (tc)", min_value=0.0, max_value=1000.0, value=180.0)
    s2 = st.number_input("Low-density lipoproteins (ldl)", min_value=0.0, max_value=500.0, value=100.0)
    s3 = st.number_input("High-density lipoproteins (hdl)", min_value=0.0, max_value=500.0, value=30.0)
    s4 = st.number_input("Total cholesterol / HDL (tch)", min_value=0.0, max_value=500.0, value=40.0)
    s5 = st.number_input("Possibly log of serum triglycerides level (ltg)", min_value=0.0, max_value=500.0, value=50.0)
    s6 = st.number_input("Blood sugar level (glu)", min_value=0.0, max_value=500.0, value=80.0)
    
    if st.button("Predict"):
        user_data = {
            "age" : age,
            "sex" : 0.050680 if sex=="Male" else -0.044642,
            "bmi" : bmi,
            "bp" : bp,
            "s1" : s1,
            "s2" : s2,
            "s3" : s3,
            "s4" : s4,
            "s5" : s5,
            "s6" : s6
        }
        
        prediction = predict_data(user_data)
        st.success(f"Your prediction result : {prediction}")
        
        user_data["target"] = prediction
        
        user_data = {
            key: int(value) if isinstance(value, (np.integer, np.int32, np.int64)) else
                 float(value) if isinstance(value, (np.floating, np.float32, np.float64, np.ndarray)) else
                 value
                 
                 for key, value in user_data.items()
        }
        collection.insert_one(user_data)
    
if __name__ == "__main__":
    main()
