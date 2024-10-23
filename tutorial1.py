import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

"""
Step 4 - Log the model and its metadata to MLflow
In this next step, we’re going to use the model that we trained, the hyperparameters that we specified for the model’s fit, and the loss metrics that were calculated by evaluating the model’s performance on the test data to log to MLflow.

The steps that we will take are:

Initiate an MLflow run context to start a new run that we will log the model and metadata to.

Log model parameters and performance metrics.

Tag the run for easy retrieval.

Register the model in the MLflow Model Registry while logging (saving) the model.

"""


# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Start an MLflow run
with mlflow.start_run():
    
    # Log the hyperparameters
    mlflow.log_params(params)
    
    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)
    
    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")
    
    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))
    
    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )
    
    '''
    Step 5 - Load the model as a Python Function (pyfunc) and use it for inference
    After logging the model, we can perform inference by:

            Loading the model using MLflow’s pyfunc flavor.

            Running Predict on new data using the loaded model.
    '''
    
# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names

result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

print("result---", result[:4])