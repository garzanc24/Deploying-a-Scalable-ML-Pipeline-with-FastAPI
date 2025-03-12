import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Import your custom modules
from ml.model import load_model, inference, apply_label
from ml.data import process_data

# Load the encoder and model at startup
encoder_path = "model/encoder.pkl"
model_path = "model/model.pkl"

# Load encoder and label binarizer
with open(encoder_path, 'rb') as f:
    encoder, lb = pickle.load(f)

# Load model
model = load_model(model_path)

# Define the data model for input validation
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# Initialize FastAPI app
app = FastAPI(
    title="Census Income Prediction API",
    description="Predicts whether income exceeds $50K/yr based on census data",
    version="1.0.0"
)

# Define categorical features
CAT_FEATURES = [
    "workclass", "education", "marital-status", "occupation", 
    "relationship", "race", "sex", "native-country"
]

# Root endpoint
@app.get("/")
async def get_root():
    """ Welcome endpoint """
    return {"greeting": "Welcome to the Census Income Prediction API!"}

# Prediction endpoint
@app.post("/data/")
async def post_inference(data: Data):
    """ Predict income based on input data """
    # Convert Pydantic model to dictionary
    data_dict = data.dict(by_alias=True)
    
    # Convert to DataFrame
    data = pd.DataFrame.from_dict({k: [v] for k, v in data_dict.items()})
    
    # Process the data
    X_processed, _, _, _ = process_data(
        data, 
        categorical_features=CAT_FEATURES, 
        training=False, 
        encoder=encoder,
        lb=lb
    )
    
    # Run inference
    prediction = inference(model, X_processed)
    
    # Apply label
    result = apply_label(prediction)
    
    return {"result": result}