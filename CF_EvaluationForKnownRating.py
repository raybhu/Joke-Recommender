import os
import json
import pandas as pd
import numpy as np
import math
from bs4 import BeautifulSoup, Comment
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


def top_n_accuracy(preds, truths, n):
    n_more_than_zero = np.where(truths > 0)[0]
    if n_more_than_zero.size < n:
        n = n_more_than_zero.size
    t_max_n = np.argpartition(truths, -n)[-n:]
    p_max_n = np.argpartition(preds, -n)[-n:]
    # print(t_max_n, p_max_n)
    intersection = np.intersect1d(p_max_n, t_max_n)
    intersection_number = intersection.size
    for i in intersection:
        if intersection_number == 0:
            break
        if preds[i] == 0:
            intersection_number -= 1
    return float(intersection_number)/10


dataset = pd.read_excel(
    os.path.abspath(
        os.path.dirname(__file__)+'/CF_ResultofKnownPrediction.xlsx/'), sheet_name=0,  usecols=range(1, 101))
dataset_test = pd.read_excel(
    os.path.abspath(
        os.path.dirname(__file__)+'/CF_ResultofKnownPrediction.xlsx/'), sheet_name=1,  usecols=range(1, 101))
print(dataset, dataset_test)
MSE = mean_squared_error(dataset_test, dataset)
RMSE = round(math.sqrt(MSE), 3)
print('MSE:%f, RMSE: %f' % (MSE, RMSE))

for rIndex, row in dataset.iterrows():
    top_10_accuracy = top_n_accuracy(
        dataset_test.iloc[rIndex].to_numpy(), row.to_numpy(), 10)
    print('The accuracy value of the row %d is %s.' %
          (rIndex+1, top_10_accuracy))
