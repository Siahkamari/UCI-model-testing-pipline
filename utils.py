import numpy as np
import time
from load_data import load_data

def normalize_XX(X_train, X_test):
    mu = np.mean(X_train, axis=0)
    sigma = np.mean(np.abs(X_train - mu), axis=0)
    sigma = np.maximum(sigma, 1e-5)
    X_train = (X_train - mu)/sigma
    X_test = (X_test - mu)/sigma
    return X_train, X_test

def evaluate(model, y_train, X_train, y_test, X_test):
  X_train, X_test = normalize_XX(X_train, X_test)

  t1 = time.perf_counter()
  model.fit(X_train, y_train)
  t2 = time.perf_counter()

  score_test = model.score(X_test, y_test)
  score_train = model.score(X_train, y_train)
  return score_train, score_test, t2 - t1


def cross_validate(model, y, X, n_folds=5):
  n = X.shape[0]

  # Permute the rows of X and y
  rng = np.random.default_rng(0)
  rp = rng.permutation(n)

  # Initializing scores
  score_test = np.zeros(n_folds)
  score_train = np.zeros(n_folds)
  elapsed_time = np.zeros(n_folds)

  for i in range(n_folds):
    # splitting the data to test and train
    test_start = int(n/n_folds * i)
    test_end = int(n/n_folds * (i+1))

    I_test = [i for i in range(test_start, test_end)]
    I_train = [i for i in range(test_start)] + [i for i in range(test_end, n)] 
    I_test = rp[I_test]
    I_train = rp[I_train]

    X_train = np.copy(X[I_train])
    X_test = np.copy(X[I_test])
    y_train = np.copy(y[I_train])
    y_test = np.copy(y[I_test])

    score_train[i], score_test[i], elapsed_time[i]  =\
      evaluate(model, y_train, X_train, y_test, X_test)

  return score_train, score_test, elapsed_time


def test(data_name, model_list, n_folds=5):
  print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%', data_name,'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
  data = load_data()

  my_data = getattr(data, data_name)()
  if len(my_data) == 4:
    y_train_all, X_train, y_test_all, X_test = my_data
    n_target = 1 if len(y_train_all.shape)==1 else y_train_all.shape[1] 
    n,dim = X_train.shape
  elif len(my_data) == 2:
    y_all, X  = my_data
    n_target = 1 if len(y_all.shape)==1 else y_all.shape[1] 
    n,dim = X.shape

  print("size of this problem (n,dim) = ",n, "x", dim)

  for target_id in range(n_target):
    print("~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~")
    print("Results for target variable ", target_id+1, "/", n_target)
    print("~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~")

    if len(my_data) == 4:
      if n_target >1:
        y_train = y_train_all[:,target_id].reshape(-1)
        y_test = y_test_all[:,target_id].reshape(-1)
      else:
        y_train = y_train_all
        y_test = y_test_all
    elif len(my_data) == 2:
      if n_target >1:
        y = y_all[:,target_id].reshape(-1)
      else:
        y = y_all
      
    # Other models
    for i, model in enumerate(model_list):
      print("############# model",i, "#############")
      if len(my_data) == 4:
        score_train, score_test, elapsed_time = evaluate(model, y_train, X_train, y_test, X_test)
      elif len(my_data)==2:
        score_train, score_test, elapsed_time = cross_validate(model, y, X, n_folds=n_folds)
      
      print("training score =", "{:.3f}".format(np.mean(score_train)))
      print("test score =", "{:.3f}".format(np.mean(score_test)))
      print('elapsed time = ', "{:.2f}".format(np.mean(elapsed_time)), 'seconds')

    
