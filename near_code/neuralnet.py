import os

# os.environ["OPENBLAS_NUM_THREADS"] = "8"

# import numpy as np
# import pandas as pd
# from autoPyTorch.api.tabular_classification import TabularClassificationTask
# import torch
# # data and metric imports
# import sklearn.model_selection
# import sklearn.datasets
# import sklearn.metrics

import numpy as np
import pickle

from autokeras import StructuredDataClassifier, StructuredDataRegressor


if False:
    X_train = np.load('data/train_X.npy')
    Y_train = np.load('data/train_Y.npy')
    X_test = np.load('data/test_X.npy')
    Y_test = np.load('data/test_Y.npy')
else:
    X_train = np.load('data/datacarlotrain_X.npy')
    Y_train = np.load('data/datacarlotrain_Y.npy')[:,1]
    Y_train[Y_train<-2] = -3
    uniques = (np.unique(Y_train)).tolist()
    Y_train = np.array([uniques.index(y) for y in Y_train])[:,None]
    X_test = np.load('data/datacarlotest_X.npy')
    Y_test = np.load('data/datacarlotest_Y.npy')[:,1]
    Y_test[Y_test<-2] = -3
    Y_test = np.array([uniques.index(y) for y in Y_test])[:,None]


clf = StructuredDataClassifier(max_trials=20, project_name='experiments/structured_data_classifier')

clf.fit(x=X_train, y=Y_train, epochs=20)

print(clf.evaluate(X_test, Y_test))

best_model = clf.export_model()
best_model.summary()