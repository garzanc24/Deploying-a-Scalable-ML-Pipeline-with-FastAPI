# Script to train machine learning model.
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, performance_on_categorical_slice

# Add code to load in the data.
data = pd.read_csv("data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

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

# Process the train data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

# Save the model and encoder
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
    print("Model saved to model/model.pkl")

with open('model/encoder.pkl', 'wb') as f:
    pickle.dump((encoder, lb), f)
    print("Model saved to model/encoder.pkl")

# Load model for inference
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)
    print("Loading model from model/model.pkl")

# Run inference on test data
preds = inference(model, X_test)

# Compute metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")

# Compute and save performance on slices
with open('slice_output.txt', 'w') as f:
    f.write(f"Overall Metrics\n")
    f.write(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}\n\n")
    
    # For each categorical feature, compute performance on slices
    for feature in cat_features:
        f.write(f"Performance on {feature} slices:\n")
        slice_metrics = performance_on_categorical_slice(
            test, model, encoder, lb, feature, cat_features
        )
        
        for value, metrics in slice_metrics.items():
            value_count = test[test[feature] == value].shape[0]
            precision, recall, fbeta = metrics
            f.write(f"{feature}: {value}, Count: {value_count}\n")
            f.write(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}\n")
        
        f.write("\n")  # Add an empty line between features