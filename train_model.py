import pandas as pd
import os
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, save_model, inference

# Load the data
data = pd.read_csv("data/clean_census.csv")

# Define categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the data
X_train, y_train, encoder, lb = process_data(
    data,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Train the model
model = train_model(X_train, y_train)

# Get model predictions
preds = inference(model, X_train)

# Compute metrics
precision, recall, fbeta = compute_model_metrics(y_train, preds)
print(f"Model performance on training data:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 score: {fbeta:.4f}")

# Save the model and encoders
os.makedirs("model", exist_ok=True)
save_model(model, "model/model.pkl")
save_model(encoder, "model/encoder.pkl")
save_model(lb, "model/lb.pkl")

print("Model, encoder, and label binarizer saved to model/ directory")