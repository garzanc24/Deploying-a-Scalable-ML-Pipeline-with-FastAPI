import os
import sys
import numpy as np
import pandas as pd
import pickle

# Add the current directory to Python path
sys.path.append(os.getcwd())

# Import your custom modules
from ml.model import load_model, inference, apply_label
from ml.data import process_data

def check_model_loading():
    # Check encoder path
    encoder_path = "model/encoder.pkl"
    model_path = "model/model.pkl"

    print("Checking file existence:")
    print(f"Encoder file exists: {os.path.exists(encoder_path)}")
    print(f"Model file exists: {os.path.exists(model_path)}")

    try:
        # Manually load and inspect the files
        print("\nInspecting encoder file:")
        with open(encoder_path, 'rb') as f:
            encoder_tuple = pickle.load(f)
        
        # Ensure we have a tuple
        if not isinstance(encoder_tuple, tuple):
            raise ValueError("Encoder file does not contain a tuple")
        
        # Unpack the tuple
        encoder, lb = encoder_tuple
        
        print("Encoder type:", type(encoder))
        print("Label Binarizer type:", type(lb))

        print("\nLoading model...")
        model = load_model(model_path)
        print(f"Model type: {type(model)}")

        # Prepare sample data for testing
        sample_data = {
            "age": [37],
            "workclass": ["Private"],
            "fnlgt": [178356],
            "education": ["HS-grad"],
            "education-num": [10],
            "marital-status": ["Married-civ-spouse"],
            "occupation": ["Prof-specialty"],
            "relationship": ["Husband"],
            "race": ["White"],
            "sex": ["Male"],
            "capital-gain": [0],
            "capital-loss": [0],
            "hours-per-week": [40],
            "native-country": ["United-States"]
        }
        sample_df = pd.DataFrame(sample_data)

        # Define categorical features
        cat_features = [
            "workclass", "education", "marital-status", "occupation", 
            "relationship", "race", "sex", "native-country"
        ]

        # Verify encoder capabilities
        print("\nEncoder verification:")
        print("Has transform method:", hasattr(encoder, 'transform'))
        
        # Process the data
        # Modify the process_data call to explicitly pass encoder and lb
        processed_data = process_data(
            sample_df, 
            categorical_features=cat_features, 
            training=False, 
            encoder=encoder,
            lb=lb
        )

        # Unpack the processed data
        X_processed, _, processed_encoder, processed_lb = processed_data

        print("Data processed successfully.")
        print(f"Processed data shape: {X_processed.shape}")

        # Run inference
        print("\nRunning model inference...")
        prediction = inference(model, X_processed)
        
        # Apply label
        result = apply_label(prediction)
        
        print("\nInference successful:")
        print(f"Prediction: {result}")

    except Exception as e:
        print(f"\nError during model checking: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_model_loading()