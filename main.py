from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import json
from datetime import datetime

app = FastAPI()

# Add CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. For production, specify allowed origins.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Load model and encoders
model = joblib.load("fertilizer_model.pkl")
soil_encoder = joblib.load("soil_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
fertilizer_encoder = joblib.load("fertilizer_encoder.pkl")

# Request schema
class FertilizerRequest(BaseModel):
    temperature: float
    humidity: float
    moisture: float
    soil_type: str
    crop_type: str
    nitrogen: float
    phosphorous: float
    potassium: float

@app.post("/predict")
def predict(data: FertilizerRequest):
    try:
        # Encode soil and crop types
        soil_encoded = soil_encoder.transform([data.soil_type])[0]
        crop_encoded = crop_encoder.transform([data.crop_type])[0]

        # Build feature array
        features = [[
            data.temperature,
            data.humidity,
            data.moisture,
            soil_encoded,
            crop_encoded,
            data.nitrogen,
            data.potassium,
            data.phosphorous
        ]]

        # Predict
        prediction_encoded = model.predict(features)[0]
        prediction_label = fertilizer_encoder.inverse_transform([prediction_encoded])[0]

        # Store the recommendation with timestamp
        recommendation = {
            "recommended_fertilizer": prediction_label,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Write the recommendation to a local file
        with open("recommendations_log.json", "a") as log_file:
            log_file.write(json.dumps(recommendation) + "\n")

        return { "recommended_fertilizer": prediction_label, "timestamp": recommendation["timestamp"] }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
