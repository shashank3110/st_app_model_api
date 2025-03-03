import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd

def fetch_raw_data():
    X,y = load_breast_cancer(return_X_y=True)

    # X,y = resample(X,y, n_samples=100000, stratify=y)
    # X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.3)
    
    
    return X,y

button = st.button("fetch data")

if button:
    X, y = fetch_raw_data()
    st.bar_chart(pd.DataFrame(y,columns=["target"]))

