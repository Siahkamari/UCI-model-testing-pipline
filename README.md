# UCI-model-testing-pipline
This is a repository for testing regression and classification models based on UCI datasets


# Prerequisite
python3
import pandas as pd
import urllib.request
import numpy as np
from IPython.display import display
from zipfile import ZipFile
import time
from sklearn.model_selection import GridSearchCV

# Availble in this repository is two classes
## 1. load_data(dataname)
Given name a of a UCI dataset from a specified list, this classs downloads, processes and provides the dataset in ($X, y$) in numpy format.
## 2. test(data_name, model_list, n_folds=5)
Given the data_name and a list of trainable models (needs to perform model.fit(X,y) and model.score(X,y)), it will evaluate each model in the model_list on the data. If the data doesn't have a specific train/test split, it will use cross validation with $n_folds$ folds and prints the average test scores.
