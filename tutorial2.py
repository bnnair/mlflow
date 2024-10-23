### Run a hyperparameter sweep
###This example tries to optimize the RMSE metric of a Keras deep learning model on a wine quality dataset. 
# It has two hyperparameters that it tries to optimize: learning_rate and momentum. We will use the Hyperopt 
# library to run a hyperparameter sweep across different values of learning_rate and momentum and record the 
# results in MLflow.

## Before running the hyperparameter sweep, let’s set the MLFLOW_TRACKING_URI environment variable to the 
# URI of our MLflow tracking server:

#   export MLFLOW_TRACKING_URI=http://localhost:5000

import keras
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models import infer_signature


