import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.
    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Initialize Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def inference(model, X):
    """ Run model inferences and return the predictions.
    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def save_model(model, path):
    """ Serializes model to a file.
    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    """ Loads pickle file from `path` and returns it."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def performance_on_categorical_slice(data, model, encoder, lb, slice_feature, cat_features):
    """
    Compute performance metrics on slices of the data based on categorical features.
    
    Inputs:
    - data: Pandas DataFrame containing the data
    - model: Trained machine learning model
    - encoder: Trained OneHotEncoder
    - lb: Trained LabelBinarizer
    - slice_feature: Feature to slice on
    - cat_features: List of categorical features
    
    Returns:
    - Dictionary with metrics for each unique value in the slice feature
    """
    slice_metrics = {}
    
    # Get unique values for the slice feature
    unique_values = data[slice_feature].unique()
    
    for value in unique_values:
        # Filter data for this slice value
        slice_data = data[data[slice_feature] == value]
        
        if len(slice_data) == 0:
            continue
        
        # Process slice data
        X_slice, y_slice, _, _ = process_data(
            slice_data, 
            categorical_features=cat_features, 
            label="salary", 
            training=False,
            encoder=encoder, 
            lb=lb
        )
        
        # Get predictions on slice
        preds = inference(model, X_slice)
        
        # Calculate metrics
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        
        # Store metrics
        slice_metrics[value] = (precision, recall, fbeta)
    
    return slice_metrics