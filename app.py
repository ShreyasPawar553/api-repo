import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pydantic import BaseModel

app = FastAPI()

# ================== PLANT DISEASE DETECTION ==================

# Paths for the TensorFlow Lite model and class indices
MODEL_PATH = "model.tflite"
CLASS_INDICES_PATH = "class_indices.json"

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Load class indices
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
else:
    class_indices = {}

# Image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    img = Image.open(BytesIO(image)).resize(target_size)
    img_array = np.expand_dims(np.array(img).astype('float32') / 255.0, axis=0)
    return img_array

# Prediction function using TensorFlow Lite
def predict_plant_disease(image):
    preprocessed_img = preprocess_image(image)
    
    # Set up the input and output tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor data
    interpreter.set_tensor(input_details[0]['index'], preprocessed_img)

    # Run inference
    interpreter.invoke()

    # Get the output
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_indices.get(str(predicted_class_index), "Unknown Class")


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        prediction = predict_plant_disease(image_data)
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


# ================== CROP & FERTILIZER RECOMMENDATION ==================

# Paths for models and scalers
FERTILIZER_MODEL_PATH = "classifier.pkl"
FERTILIZER_INFO_PATH = "fertilizer.pkl"
CROP_MODEL_PATH = "model.pkl"
SCALER_STANDARD_PATH = "standscaler.pkl"
SCALER_MINMAX_PATH = "minmaxscaler.pkl"

# Load models and scalers
fertilizer_model = pickle.load(open(FERTILIZER_MODEL_PATH, 'rb'))
fertilizer_info = pickle.load(open(FERTILIZER_INFO_PATH, 'rb'))
crop_model = pickle.load(open(CROP_MODEL_PATH, 'rb'))
scaler_standard = pickle.load(open(SCALER_STANDARD_PATH, 'rb'))
scaler_minmax = pickle.load(open(SCALER_MINMAX_PATH, 'rb'))

# Input schemas for the APIs
class FertilizerData(BaseModel):
    temp: float
    humi: float
    mois: float
    soil: float
    crop: int
    nitro: float
    pota: float
    phosp: float

class CropData(BaseModel):
    N: float
    P: float
    K: float
    temp: float
    humidity: float
    ph: float
    rainfall: float


@app.post("/fertilizer/predict")
async def predict_fertilizer(data: FertilizerData):
    try:
        input_data = np.array([[data.temp, data.humi, data.mois, data.soil, data.crop, data.nitro, data.pota, data.phosp]])
        prediction_idx = fertilizer_model.predict(input_data)[0]
        result_label = fertilizer_info.classes_[prediction_idx] if hasattr(fertilizer_info, 'classes_') else 'Unknown'
        return JSONResponse(content={"fertilizer": result_label})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post("/crop/predict")
async def predict_crop(data: CropData):
    try:
        feature_list = np.array([[data.N, data.P, data.K, data.temp, data.humidity, data.ph, data.rainfall]])
        scaled_features = scaler_minmax.transform(feature_list)
        final_features = scaler_standard.transform(scaled_features)
        prediction = crop_model.predict(final_features)[0]
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
            7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
            12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
            17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
            21: "Chickpea", 22: "Coffee"
        }
        result = crop_dict.get(int(prediction), 'Unknown crop')
        return JSONResponse(content={"recommended_crop": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
