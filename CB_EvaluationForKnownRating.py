import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
dataset = pd.read_excel(
    os.path.abspath(
        os.path.dirname(__file__)+'/CB_ResultofBinaryPrediction.xlsx/'), sheet_name=0,  usecols=range(1, 101))
dataset_test = pd.read_excel(
    os.path.abspath(
        os.path.dirname(__file__)+'/CB_ResultofBinaryPrediction.xlsx/'), sheet_name=1,  usecols=range(1, 101))
print(dataset, dataset_test)
result = {'accuracy_score': [], 'f1_score': [],
          'recall_score': [], 'confusion matrix': []}
for rIndex, row in dataset.iterrows():
    deletedLabel = set()
    for clabel, value in row.iteritems():
        if float(value) == 0:
            deletedLabel.add(clabel)
    # print(type(row))
    for tclabel, tvalue in dataset_test.iloc[rIndex, :].iteritems():
        if float(tvalue) == 0:
            deletedLabel.add(tclabel)
    deletedLabel = list(deletedLabel)
    processedOriginalRow = row.drop(labels=deletedLabel)
    processedPredictedRow = dataset_test.iloc[rIndex, :].drop(
        labels=deletedLabel)
    print('accuracy_score: ', accuracy_score(
        processedOriginalRow, processedPredictedRow))
    print('f1_score: ', f1_score(processedOriginalRow, processedPredictedRow))
    print('recall_score: ', recall_score(
        processedOriginalRow, processedPredictedRow))
    print('confusion matrix: \n', confusion_matrix(
        processedOriginalRow, processedPredictedRow))
    result['accuracy_score'].append(accuracy_score(
        processedOriginalRow, processedPredictedRow))
    result['f1_score'].append(
        f1_score(processedOriginalRow, processedPredictedRow))
    result['recall_score'].append(recall_score(
        processedOriginalRow, processedPredictedRow))
    result['confusion matrix'].append(
        confusion_matrix(
            processedOriginalRow, processedPredictedRow))
result = pd.DataFrame(result, index=range(
    1, len(result['accuracy_score'])+1))
result.to_excel("CB_EvaluationforBinaryPrediction.xlsx",  sheet_name='Sheet1')
