import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

@pytest.fixture
def data():
    """
    Get a small sample dataset for testing
    """
    # Either load a small subset of your data or create a synthetic dataset
    df = pd.read_csv("data/census.csv")
    return df.sample(100, random_state=42)  # Small sample for testing

def test_process_data(data):
    """
    Test process_data function returns the expected types and shapes
    """
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
    
    X_train, y_train, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    
    # Test output types
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert y_train.shape[0] == X_train.shape[0]  # Same number of samples
    
    # Process again to test with existing encoder
    X_test, y_test, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", 
        training=False, encoder=encoder, lb=lb
    )
    
    # Verify same shape features after encoding
    assert X_test.shape[1] == X_train.shape[1]

def test_train_model():
    """
    Test train_model returns the expected model type
    """
    # Create some dummy data
    X = np.random.random((100, 10))
    y = np.random.randint(0, 2, 100)
    
    model = train_model(X, y)
    
    # Check model type - assuming it's a RandomForestClassifier, change if different
    assert isinstance(model, RandomForestClassifier)
    
    # Verify model was fitted
    assert hasattr(model, 'classes_')
    assert len(model.classes_) > 0

def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns expected values for known inputs
    """
    # Create known ground truth and predictions
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    preds = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    
    # Expected values calculated by hand
    # Precision: TP/(TP+FP) = 2/(2+2) = 0.5
    # Recall: TP/(TP+FN) = 2/(2+2) = 0.5
    # F1: 2*precision*recall/(precision+recall) = 2*0.5*0.5/(0.5+0.5) = 0.5
    
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    assert np.isclose(precision, 0.5)
    assert np.isclose(recall, 0.5)
    assert np.isclose(fbeta, 0.5)
    
    # Test with perfect predictions
    precision, recall, fbeta = compute_model_metrics(y, y)
    assert np.isclose(precision, 1.0)
    assert np.isclose(recall, 1.0)
    assert np.isclose(fbeta, 1.0)

def test_inference():
    """
    Test that inference function returns predictions of the expected shape and type
    """
    # Load the model
    if os.path.exists('model/model.pkl'):
        with open('model/model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        # Create some dummy data matching the expected feature dimensions
        # Note: You'll need to adjust the feature count to match your model
        feature_count = 108  # Replace with your actual feature count
        X = np.random.random((10, feature_count))
        
        # Run inference
        preds = inference(model, X)
        
        # Check predictions shape and type
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == X.shape[0]
        assert set(np.unique(preds)).issubset({0, 1})  # Binary classification
    else:
        pytest.skip("Model file not found, skipping inference test")