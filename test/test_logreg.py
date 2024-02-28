"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import copy
import numpy as np
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier
# Ignore warnings raised by SGDClassifier
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def test_prediction():
    # Init LogisticRegressor  
    log_model = logreg.LogisticRegressor(num_feats=3, learning_rate=0.00001, 
                                         tol=0.01, max_iter=10, batch_size=10)
    # Set weights to 2
    log_model.W = np.ones(len(log_model.W))*2 
    # Set example input
    X_small = np.array([[10,8,10],[0,0,0],[-10,-8,-10]])
    X_small = np.hstack([X_small, np.ones((X_small.shape[0], 1))])  # Add bias term
    
    # Make prediction
    y_pred = log_model.make_prediction(X_small)
    # For a logistic regression model that predicts a probablility for a class
    # using the sigmoid function, and where each input feature has equal weight,
    # the prediction for large, positive features should be close to 1,
    # the prediction for large, negative features should be close to 0, and
    # the prediction for zero-value features should be in the dynamic range of (0, 1),
    # i.e., sigmoid(bias term)
    assert np.allclose(y_pred, [1, expit(2*1), 0])
    

def test_loss_function():
    # Init LogisticRegressor  
    log_model = logreg.LogisticRegressor(num_feats=3, learning_rate=0.00001, 
                                         tol=0.01, max_iter=10, batch_size=10)
    # Set weights to 2
    log_model.W = np.ones(len(log_model.W))*2 
    # Set example input
    X_small = np.array([[1,0.5,1],[1,0,0],[-1,-0.5,-1]])
    X_small = np.hstack([X_small, np.ones((X_small.shape[0], 1))])  # Add bias term
    
    # Calculate the loss
    y_pred = log_model.make_prediction(X_small)
    y_true = np.array([1, 1, 0])
    train_loss = log_model.loss_function(y_true, y_pred)
    # Use a different method to calculate the binary cross entropy
    sk_loss = log_loss(y_true, y_pred)
    assert np.allclose(train_loss, sk_loss)


@ignore_warnings(category=ConvergenceWarning)
def test_gradient():
    # Init LogisticRegressor  
    log_model = logreg.LogisticRegressor(num_feats=3, learning_rate=0.01, 
                                         tol=0.01, max_iter=10, batch_size=10)
    # Set weights to 2
    log_model.W = np.ones(len(log_model.W))*2
    # Set example input
    X_small = np.array([[1,0.5,1],[1,-1,1]])
    X_small = np.hstack([X_small, np.ones((X_small.shape[0], 1))])  # Add bias term
    
    # Calculate the gradient.
    y_true = np.array([1,0])
    grad = log_model.calculate_gradient(y_true, X_small)
    # Use sklearn's Stochastic Gradient Descent Classifier to check the calculated gradient.
    # sklearn has no Gradient Descent Classifier, but parameters for SGDClassifier are set 
    # such that the weights, coef_, become equivalent to Gradient Descent's updated weights 
    # after 1 iteration:
    # Set the loss to be binary cross entropy.
    # Set penalty to None to prevent regularization.
    # Set the max_iter=1 since we are checking the gradient after one pass.
    # Set the learning rate to be the same as in log_model.
    # Set average=1. Unlike GD, SGD updates weights for a single training example at a time.
    # The average parameter sets coef_ to the average weight across all updates. Here, I
    # impose averaging to start after seeing 1 sample, so that the coef_ after all updates
    # in 1 iteration is akin to getting the mean gradient across all samples.
    # That is, the gradient calculated from coef_ will be almost equal to that of GD.
    sk_model = SGDClassifier(loss="log_loss", penalty=None, max_iter=1, 
                             learning_rate='constant', eta0=0.01, average=1)
    # Use the same initial weights as log_model.
    sk_model.fit(X_small, y_true, coef_init=np.array([2,2,2,2]))
    # Get the gradient from coef_. grad = (updated_W - prev_W) / -learning rate
    sk_grad = (sk_model.coef_ - np.array([2,2,2,2])) / -0.01
    assert np.allclose(grad, sk_grad, atol=1e-03)

    
def test_training():
	# Load data
    X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
    )

    # Scale the data, since values vary across feature. Note that we
    # fit on the training data and use the same scaler for X_val.
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    # Test LogisticRegressor prediction using optimized hyperparameters
    np.random.seed(42)
    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.001, tol=0.001, max_iter=100, batch_size=50)
    init_weights = copy.deepcopy(log_model.W)
    log_model.train_model(X_train, y_train, X_val, y_val)
    
    # Check that weights have updated
    assert not np.allclose(log_model.W, init_weights)
    
    # Check that the model performs better than chance.
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    assert roc_auc_score(y_val, log_model.make_prediction(X_val)) > 0.5
    