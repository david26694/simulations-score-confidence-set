import numpy as np
from doubleml import DoubleMLData, DoubleMLIIVM
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression


class DummyLogisticRegression(LogisticRegression):
    """Custom LogisticRegression that always predicts 0.5"""
    
    def predict(self, X):
        return np.full(X.shape[0], 0.5)


def fit_dml_model(data):
    """
    Fit a double machine learning model
    
    Args:
        data: Dictionary containing the data
        
    Returns:
        Fitted DoubleMLIIVM model
    """
    # Unpack data
    Y = data["Y"]
    Z = data["Z"]
    A = data["A"]
    X = data["X"]

    # Prepare DoubleMLData object
    dml_data = DoubleMLData.from_arrays(x=X, y=Y, d=A, z=Z)

    # Specify machine learning methods for nuisance functions
    learner_g = RandomForestRegressor()
    classifier_m = DummyLogisticRegression()
    classifier_r = RandomForestClassifier()
    dml_model = DoubleMLIIVM(dml_data, ml_g=learner_g, ml_m=classifier_m, ml_r=classifier_r, n_folds=2)
    dml_model.fit()
    return dml_model