from io import BytesIO
import json
from fastapi import FastAPI, File, UploadFile
import uvicorn 
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import requests
import sys
sys.path.append('../')
from WasteManagementApplication.config.definitions import ROOT_DIR
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:65000",

]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#MODEL = tf.keras.models.load_model(os.path.join(ROOT_DIR, 'saved_models', '1'))

endpoint = "http://localhost:8501/v1/models/waste_separation:predict"
@app.get("/ping")

async def ping():
    return "Hello buddy"

def read_file_as_image(data) -> np.ndarray:
    img = Image.open(BytesIO(data))
    img = img.resize((244,244))
    image = np.array(img)
    image = image/255
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    #prediction = MODEL.predict(img_batch)
    
    json_data = {    "instances": img_batch.tolist() }
    response = requests.post(endpoint, json=json_data)
    
    
    prediction = np.array(response.json()["predictions"][0][0])
    prediction
    if prediction > 0.5:
        predicted_class = 'Recyclable Waste' 
        confidence = np.round(prediction, 2)
    else:
        predicted_class = 'Organic Waste'
        confidence = np.round((1-prediction), 2)
 
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
        

if __name__  ==  "__main__":
    uvicorn.run(app, host='localhost', port=65000)  