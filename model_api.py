from typing import Union, List, Dict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score
from fastapi import FastAPI, HTTPException
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from pydantic import BaseModel
import json
import asyncio

app = FastAPI()

async def fetch_data():
    X,y = load_breast_cancer(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.3)
    
    
    return X_train, X_val, y_train, y_val

class Payload(BaseModel):
    model: str
    params: Dict
    data: Dict

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/train")
async def train_model(payload:Payload): 

    model = payload.model
    params = payload.params
    data = payload.data # client side data

    try:
        model_dict = {"RandomForest":RandomForestClassifier, "ExtraTrees":ExtraTreesClassifier}

        # server side data
        # data = await fetch_data()
        # X_train, X_val, y_train, y_val = data

        # client side data
        X_train, X_val, y_train, y_val = data["X_train"], data["X_val"], data["y_train"], data["y_val"]

        print(params)
        if model not in model_dict:
            raise HTTPException(status_code=500, detail="Invalid model name")
        model_obj = model_dict[model](**params)
        model_obj.fit(X_train,y_train)
        y_pred = model_obj.predict(X_val)
        eval = f1_score(y_val,y_pred)



        
    except Exception as e:
        raise HTTPException(detail=str(e), status_code=500,)

    return {"model":model, "params":params, "val_metric": eval}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


