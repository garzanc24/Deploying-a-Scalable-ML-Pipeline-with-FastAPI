import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label].values  # Convert to NumPy array immediately
        X = X.drop(columns=[label], axis=1)
    else:
        y = np.array([])

    # Handle categorical features
    if categorical_features:
        X_categorical = X[categorical_features].values
        X_continuous = X.drop(columns=categorical_features, axis=1)
    else:
        X_categorical = np.array([])  # Empty array if no categorical features exist
        X_continuous = X

    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        if categorical_features:
            X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y).ravel()
    else:
        if categorical_features:
            X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y).ravel()
        except AttributeError:
            pass  # If y is empty during inference

    # Merge continuous and categorical data
    if categorical_features:
        X = np.concatenate([X_continuous, X_categorical], axis=1)
    else:
        X = X_continuous.to_numpy()

    return X, y, encoder, lb


def apply_label(inference):
    """ Convert the binary label in a single inference sample into string output. """
    return ">50K" if inference[0] == 1 else "<=50K"
