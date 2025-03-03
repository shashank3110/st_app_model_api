import streamlit as st
import requests
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
st.title("Model Playground")

def fetch_data():
    X,y = load_breast_cancer(return_X_y=True)
    # st.write(type(X))
    # st.write(resample(n_samples=10000, stratify=y))
    X,y = resample(X,y, n_samples=1000, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.3)
    
    
    return X_train, X_val, y_train, y_val


model = st.selectbox(label="select Model",options=[None,'RandomForest', 'ExtraTrees', 'XGBoost'])

if model:
    #{"X_train":X_train,"y_train":y_train, "X_val":X_val,"y_val":y_val}
    X_train, X_val, y_train, y_val =  fetch_data()
    print(X_train.shape)

    # print(X_train.shape, len(X_train.tolist()))
                                                                            #    "data": [X_train.tolist(), X_val.tolist(), y_train.tolist(), y_val.tolist()], 

    

    # data at server side
    # response =  requests.post(f'http://127.0.0.1:8000/train', json={"model":model,
    #                                                               "params":{"n_estimators":100}})

    # data at server side
    response =  requests.post(f'http://127.0.0.1:8000/train', json={"model":model,
                                                                  "params":{"n_estimators":300, "random_state":42},
                                                                  "data":{"X_train":X_train.tolist(), "X_val":X_val.tolist(),"y_train":y_train.tolist(),"y_val":y_val.tolist()}
                                                                  })
    st.write(response.text)
    st.write(response)