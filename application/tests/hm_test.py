import warnings
from typing import Tuple, Union, List

import flwr as fl
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from application.src.clientbuilder import FlowerClientTrainingBuilder

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(
        model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.
    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 1  # MNIST has 10 classes
    n_features = 10  # Number of features in dataset
    model.classes_ = np.array([i for i in range(10)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))
    return model


class HMTestClient(fl.client.NumPyClient):

    def __init__(self, training_id, configuration):
        self.training_id = training_id
        self.configuration = configuration
        train = sns.load_dataset('titanic')

        # Split train set into 10 partitions and randomly use one for training.
        train['fare'].hist(bins=40, figsize=(10, 4))

        train.groupby('pclass').mean()['age'].round()

        mean_class1 = train.groupby('pclass').mean()['age'].round().loc[1]
        mean_class2 = train.groupby('pclass').mean()['age'].round().loc[2]
        mean_class3 = train.groupby('pclass').mean()['age'].round().loc[3]

        train.loc[train['pclass'] == 1, 'age'] = train.loc[train['pclass']
                                                           == 1, 'age'].fillna(value=mean_class1)
        train.loc[train['pclass'] == 2, 'age'] = train.loc[train['pclass']
                                                           == 2, 'age'].fillna(value=mean_class2)
        train.loc[train['pclass'] == 3, 'age'] = train.loc[train['pclass']
                                                           == 3, 'age'].fillna(value=mean_class3)

        train.drop(['embark_town', 'who', 'class',
                   'deck', 'alive'], axis=1, inplace=True)

        # dropping the 1 missing value in Embarked column
        train.dropna(inplace=True)

        # I will now convert some of the categorical features in the dataset into dummy variables that our machine learning model can accept.

        sex = pd.get_dummies(train['sex'], drop_first=True)

        embark = pd.get_dummies(train['embarked'], drop_first=True)

        train = pd.concat([train, sex, embark], axis=1)

        train.drop(['sex', 'embarked'], axis=1, inplace=True)

        # Train and build Classifier

        X = train.drop('survived', axis=1)
        y = train['survived']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=101)
        # Create LogisticRegression Model
        self.model = LogisticRegression()

        # Setting initial parameters, akin to model.compile for keras models
        self.model = set_initial_params(self.model)

    def get_parameters(self, config):  # type: ignore
        return get_model_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore
        set_model_params(self.model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
        print(f"Training finished for round with config {config}")
        return get_model_parameters(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        set_model_params(self.model, parameters)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}


class HMTestBuilder(FlowerClientTrainingBuilder):

    def __init__(self, training_id, configuration):
        self.configuration = configuration
        self.client = HMTestClient(training_id, configuration)

    def prepare_training(self):
        return self.client

    def add_model(self):
        pass

    def add_optimizer(self) -> None:
        pass
