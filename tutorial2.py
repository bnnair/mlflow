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
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models import infer_signature

## Now load the dataset and split it into training, validation, and test sets.


# Load dataset
data = pd.read_csv(
    "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv",
    sep=";",
)

# Split the data into training, validation, and test sets
train, test = train_test_split(data, test_size=0.25, random_state=42)
train_x = train.drop(["quality"], axis=1).values
train_y = train[["quality"]].values.ravel()
test_x = test.drop(["quality"], axis=1).values
test_y = test[["quality"]].values.ravel()

train_x, valid_x, train_y, valid_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=42
)
print("train-x shape: ", train_x.shape)
signature = infer_signature(train_x, train_y)

### Then let’s define the model architecture and train the model. The train_model function uses 
# MLflow to track the parameters, results, and model itself of each trial as a child run

def train_model(params, epochs, train_x, train_y, valid_x, valid_y, test_x, test_y):
    # Define model architecture
    mean = np.mean(train_x, axis=0)
    var = np.var(train_x, axis=0)
    model = keras.Sequential(
        [
            keras.Input([train_x.shape[1]]),
            keras.layers.Normalization(mean=mean, variance=var),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1),
        ]
    )

    # Compile model
    model.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=params["lr"], momentum=params["momentum"]
        ),
        loss="mean_squared_error",
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    # Train model with MLflow tracking
    with mlflow.start_run(nested=True):
        model.fit(
            train_x,
            train_y,
            validation_data=(valid_x, valid_y),
            epochs=epochs,
            batch_size=64,
        )
        # Evaluate the model
        eval_result = model.evaluate(valid_x, valid_y, batch_size=64)
        eval_rmse = eval_result[1]

        # Log parameters and results
        mlflow.log_params(params)
        mlflow.log_metric("eval_rmse", eval_rmse)

        # Log model
        mlflow.tensorflow.log_model(model, "model", signature=signature)

        return {"loss": eval_rmse, "status": STATUS_OK, "model": model}
    
    
### The objective function takes in the hyperparameters and returns the results of the train_model function 
# for that set of hyperparameters.

def objective(params):
    # MLflow will track the parameters and results for each run
    result = train_model(
        params,
        epochs=5,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        test_x=test_x,
        test_y=test_y,
    )
    return result

####Next, we will define the search space for Hyperopt. In this case, we want to try different values of 
# learning-rate and momentum. Hyperopt begins its optimization process by selecting an initial set of 
# hyperparameters, typically chosen at random or based on a specified domain space. This domain space defines 
# the range and distribution of possible values for each hyperparameter. After evaluating the initial set, 
# Hyperopt uses the results to update its probabilistic model, guiding the selection of subsequent hyperparameter
# sets in a more informed manner, aiming to converge towards the optimal solution.

space = {
    "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-1)),
    "momentum": hp.uniform("momentum", 0.0, 1.0),
}

mlflow.set_experiment("/wine-quality")

with mlflow.start_run():
    # Conduct the hyperparameter search using Hyperopt
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=10,
        trials=trials,
    )

    # Fetch the details of the best run
    best_run = sorted(trials.results, key=lambda x: x["loss"])[0]

    # Log the best parameters, loss, and model
    mlflow.log_params(best)
    mlflow.log_metric("eval_rmse", best_run["loss"])
    mlflow.tensorflow.log_model(best_run["model"], "model", signature=signature)

    # Print out the best parameters and corresponding loss
    print(f"Best parameters: {best}")
    print(f"Best eval rmse: {best_run['loss']}")
    
    ## Compare the results
    #### Open the MLflow UI in your browser at the MLFLOW_TRACKING_URI. You should see a nested list of runs. 
    # In the default Table view, choose the Columns button and add the Metrics | eval_rmse column and the 
    # Parameters | lr and Parameters | momentum column. To sort by RMSE ascending, click the eval_rmse column 
    # header. The best run typically has an RMSE on the test dataset of ~0.70. You can see the parameters of the 
    # best run in the Parameters column.
    
    ## Choose Chart view. Choose the Parallel coordinates graph and configure it to show the lr and 
    # momentum coordinates and the eval_rmse metric. Each line in this graph represents a run and associates 
    # each hyperparameter evaluation run’s parameters to the evaluated error metric for the run.
    
    