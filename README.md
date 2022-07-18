# IPLcricketscoreprediction

<img width="853" alt="Screenshot 2022-07-18 at 03 57 09" src="https://user-images.githubusercontent.com/100385953/179435140-a64d574e-2495-4343-ab62-5381af19d0cd.png">


In this project we will try to predict the final score of a match if sufficient conditons are given.
As we know IPL is one of the biggest flex when it comes to Cricket all over the world. It would be very exciting to work on a Data Set which gives us the insight of the Data of different matches.
We will be doing all the Data Cleaning techniques and anlysing our data, then finally predicting our results comparing different models and then we will see how we can use Auto ML for the same. We will be using TPOT Auto ML library for the project.
Let us have a look on the TimeLine for our Project:

Importing Libraries and DataSet
Data Analysis and Cleaning
Data Preprocessing
Model Building using ML models
Model Building and Predictions using Auto ML Library i.e TPOT Library


# TPOT Library

![tpot-logo](https://user-images.githubusercontent.com/100385953/179435235-9dbaed79-2373-4f85-b3d7-308c8e977338.jpg)

TPOT stands for Tree-based Pipeline Optimization Tool. Consider TPOT your Data Science Assistant. TPOT is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.

TPOT will automate the most tedious part of machine learning by intelligently exploring thousands of possible pipelines to find the best one for your data.

<img width="986" alt="Screenshot 2022-07-18 at 03 59 22" src="https://user-images.githubusercontent.com/100385953/179435266-aa3d0695-eb4e-4cbb-89b4-26001317d1fd.png">

Once TPOT is finished searching (or you get tired of waiting), it provides you with the Python code for the best pipeline it found so you can tinker with the pipeline from there.

TPOT is built on top of scikit-learn, so all of the code it generates should look familiar... if you're familiar with scikit-learn, anyway.

TPOT is still under active development and we encourage you to check back on this repository regularly for updates.

For further information about TPOT, please see the project documentation.

License

Please see the repository license for the licensing and usage information for TPOT.

Generally, we have licensed TPOT to make it as widely usable as possible.

Installation

We maintain the TPOT installation instructions in the documentation. TPOT requires a working installation of Python.

Usage

TPOT can be used on the command line or with Python code.

Click on the corresponding links to find more information on TPOT usage in the documentation.

Examples

Classification

Below is a minimal working example with the optical recognition of handwritten digits dataset.

from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25, random_state=42)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_digits_pipeline.py')
Running this code should discover a pipeline that achieves about 98% testing accuracy, and the corresponding Python code should be exported to the tpot_digits_pipeline.py file and look similar to the following:

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.9799428471757372
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=LogisticRegression(C=0.1, dual=False, penalty="l1")),
    RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.35000000000000003, min_samples_leaf=20, min_samples_split=19, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
Regression

Similarly, TPOT can optimize pipelines for regression problems. Below is a minimal working example with the practice Boston housing prices data set.

from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

housing = load_boston()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target,
                                                    train_size=0.75, test_size=0.25, random_state=42)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')
which should result in a pipeline that achieves about 12.77 mean squared error (MSE), and the Python code in tpot_boston_pipeline.py should look similar to:

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -10.812040755234403
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    ExtraTreesRegressor(bootstrap=False, max_features=0.5, min_samples_leaf=2, min_samples_split=3, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)


