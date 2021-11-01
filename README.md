# UCI-model-testing-pipline
This is a repository for testing regression and classification models based on UCI datasets


# Prerequisites
python3

import pandas as pd 

import urllib.request

import numpy as np

from IPython.display import display

from zipfile import ZipFile

import time

from sklearn.model_selection import GridSearchCV


# Available in this repository is two classes and and example
## 1. load_data(dataname)
Given name a of a UCI dataset from a specified list, this class downloads, processes and returns the dataset in (X, y) or (X_train, y_train, X_test, y_test) with numpy format. The list of the availble datasets is in the example.ipynb. We plan to grow this list.
## 2. test(data_name, model_list, n_folds=5)
Given the data_name and a list of trainable models (needs to perform model.fit(X,y) and model.score(X,y)), it will evaluate each model in the model_list on the data. If the data doesn't have a specific train/test split, it will use cross validation with number of folds = n_folds and prints the average test scores.
## example.ipynb
An Ipython notebook that loads the datasets and tests on multiple regression and classification models. 
