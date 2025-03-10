import pytest
import numpy as np
import pandas as pd
from ml.model import train_model, inference, compute_model_metrics
from ml.data import process_data  # Correct import

@pytest.fixture
def sample_data():
    """ Provides a small dataset for testing. """
    data = {
        "age": [25, 40, 35, 50],
        "workclass": ["Private", "Self-emp", "Private", "Gov"],
        "education": ["Bachelors", "Masters", "PhD", "Bachelors"],
        "salary": [">50K", "<=50K", ">50K", "<=50K"]
    }
    df = pd.DataFrame(data)
    categorical_features = ["workclass", "education"]
    label = "salary"
    
    # Process Data
    X, y, encoder, lb = process_data(df, categorical_features, label, training=True)
    return X, y, encoder, lb

def test_train_model(sample_data):
    """ Test if the model trains successfully and returns a model. """
    X, y, _, _ = sample_data
    model = train_model(X, y)

    assert model is not None
    assert hasattr(model, "predict"), "Trained model should have a 'predict' method."

def test_inference(sample_data):
    """ Test if the inference function returns valid predictions. """
    X, y, _, _ = sample_data
    model = train_model(X, y)
    
    preds = inference(model, X)

    assert isinstance(preds, np.ndarray), "Inference output should be a NumPy array."
    assert preds.shape[0] == X.shape[0], "Number of predictions should match input data."

def test_compute_model_metrics():
    """ Test if precision, recall, and f1-score are correctly computed. """
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 1])  # One incorrect prediction

    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    assert 0 <= precision <= 1, "Precision should be between 0 and 1."
    assert 0 <= recall <= 1, "Recall should be between 0 and 1."
    assert 0 <= f1 <= 1, "F1-score should be between 0 and 1."
