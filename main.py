import click

import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

dataset_csv = "./Dataset/insurance.csv"

def get_metrics(actual, predicted):
    root_mean_square_error = np.sqrt(mean_squared_error(actual, predicted))
    mean_avg_error = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return root_mean_square_error, mean_avg_error, r2

@click.command()
@click.option("--alpha", default=0.5, type=float)
@click.option("--l1_ratio", default=0.1, type=float)
def _run_workflow(alpha, l1_ratio):
    with mlflow.start_run():
        ds = pd.read_csv(dataset_csv)
        ds = pd.get_dummies(ds)

        train, test = train_test_split(ds)

        train_x = train.drop(["charges"], axis=1)
        train_y = train[["charges"]]
        test_x = test.drop(["charges"], axis=1)
        test_y = test[["charges"]]

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(train_x, train_y)

        predicted = model.predict(test_x)
        (root_mean_square_error, mean_avg_error, r2) = get_metrics(test_y, predicted)
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("Root Mean Square Error: %s" % root_mean_square_error)
        print("Mean Average Error: %s" % mean_avg_error)
        print("R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse", root_mean_square_error)
        mlflow.log_metric("mae", mean_avg_error)

        mlflow.sklearn.log_model(model, "model")



if __name__ == "__main__":
    _run_workflow()