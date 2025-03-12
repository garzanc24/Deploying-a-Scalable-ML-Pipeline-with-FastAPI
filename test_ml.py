import pytest
import pandas as pd
import numpy as np
from ml.data import process_data, apply_label
from ml.model import train_model, inference, compute_model_metrics

@pytest.fixture
def sample_data():
    """
    Create a small sample dataset for testing
    """
    data = {
        'age': [25, 38, 45, 32, 55],
        'workclass': ['Private', 'Self-emp', 'Private', 'Federal-gov', 'Private'],
        'fnlgt': [226802, 89814, 160323, 234721, 338409],
        'education': ['HS-grad', 'Bachelors', 'Masters', 'HS-grad', 'Doctorate'],
        'education-num': [9, 13, 14, 9, 16],
        'marital-status': ['Married', 'Married', 'Divorced', 'Married', 'Never-married'],
        'occupation': ['Sales', 'Exec-managerial', 'Prof-specialty', 'Adm-clerical', 'Prof-specialty'],
        'relationship': ['Husband', 'Husband', 'Not-in-family', 'Husband', 'Not-in-family'],
        'race': ['White', 'White', 'Black', 'Asian-Pac-Islander', 'White'],
        'sex': ['Male', 'Male', 'Female', 'Male', 'Female'],
        'capital-gain': [0, 15024, 0, 0, 0],
        'capital-loss': [0, 0, 0, 0, 0],
        'hours-per-week': [40, 50, 40, 35, 60],
        'native-country': ['United-States', 'United-States', 'United-States', 'India', 'United-States'],
        'salary': ['<=50K', '>50K', '<=50K', '<=50K', '>50K']
    }
    return pd.DataFrame(data)

def test_process_data(sample_data):
    """
    Test that process_data correctly handles categorical features and creates
    the expected shape of processed data.
    """
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    
    # Test training mode
    X_train, y_train, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, 
        label="salary", training=True
    )
    
    # Check shapes
    assert X_train.shape[0] == 5, "Wrong number of samples"
    assert y_train.shape[0] == 5, "Wrong number of labels"
    
    # Check encoders are created
    assert encoder is not None, "Encoder should be created"
    assert lb is not None, "Label binarizer should be created"
    
    # Test inference mode
    X_test, y_test, _, _ = process_data(
        sample_data, categorical_features=cat_features,
        label="salary", training=False, encoder=encoder, lb=lb
    )
    
    # Check processed data is same shape in inference mode
    assert X_test.shape == X_train.shape, "Training and inference data should have same shape"

def test_train_and_inference(sample_data):
    """
    Test that model training works and produces predictions of the expected format.
    """
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    
    # Process data
    X, y, encoder, lb = process_data(
        sample_data, 
        categorical_features=cat_features,
        label="salary", 
        training=True
    )
    
    # Train model
    model = train_model(X, y)
    
    # Check model exists
    assert model is not None, "Model should be trained"
    
    # Run inference
    preds = inference(model, X)
    
    # Check predictions shape and type
    assert len(preds) == len(y), "Predictions should match number of samples"
    assert all(isinstance(pred, (int, np.integer)) for pred in preds), "Predictions should be integers"
    assert all(pred in [0, 1] for pred in preds), "Predictions should be binary"

def test_compute_metrics():
    """
    Test that metric computation works correctly with known values.
    """
    # Test case 1: Perfect predictions
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0])
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == 1.0, "Precision should be 1.0 for perfect predictions"
    assert recall == 1.0, "Recall should be 1.0 for perfect predictions"
    assert fbeta == 1.0, "F1 score should be 1.0 for perfect predictions"
    
    # Test case 2: Imperfect predictions
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 1, 1])
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
    assert 0 <= fbeta <= 1, "F1 score should be between 0 and 1"
    
    # Check specific values for this test case
    assert precision == 0.5, "Precision calculation incorrect"
    assert recall == 0.5, "Recall calculation incorrect"
    assert fbeta == 0.5, "F1 score calculation incorrect"