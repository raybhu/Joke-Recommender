import os
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, Comment
from sklearn.neighbors import NearestNeighbors
JOKE_PATH = os.path.abspath(
    os.path.dirname(__file__)+'/Dataset/jokes/')
DATASETS_PATH = {'dataset1': os.path.abspath(
    os.path.dirname(__file__)+'/Dataset/jester_dataset_1/'), 'dataset2': os.path.abspath(
    os.path.dirname(__file__)+'/Dataset/jester_dataset_2/'), 'dataset3': os.path.abspath(
    os.path.dirname(__file__)+'/Dataset/jester_dataset_3/')}

dataset_path = DATASETS_PATH['dataset1']
with pd.ExcelWriter('CF_ResultofKnownPrediction.xlsx') as writer:
    frames = []
    frames_test = []
    for f in os.listdir(dataset_path):
        # if f != 'jester-data-1.xlsx':
        #     continue
        if 'jester' in f:
            dataset = pd.read_excel(
                dataset_path+'/'+f, header=None, usecols='B:CW')
            predictedJokesLabels = {}
            knownRatingLabels = {}
            for rIndex, row in dataset.iterrows():
                # print(row)
                for clabel, value in row.iteritems():
                    if int(value) == 99:
                        dataset.at[rIndex, clabel] = 0
                        if str(rIndex) not in predictedJokesLabels:
                            predictedJokesLabels[str(rIndex)] = []
                        predictedJokesLabels[str(rIndex)].append(clabel)
                    else:
                        if str(rIndex) not in knownRatingLabels:
                            knownRatingLabels[str(rIndex)] = []
                        knownRatingLabels[str(rIndex)].append(clabel)
            dataset_test = dataset.copy()
            model_knn = NearestNeighbors(metric='cosine', algorithm='auto')
            model_knn.fit(dataset)
            maxIndex = int(dataset.shape[0]*0.1)
            # maxIndex = 10
            for rIndex, row in dataset.iterrows():
                if int(rIndex) == maxIndex:
                    break
                if str(rIndex) in knownRatingLabels:
                    distances, indices = model_knn.kneighbors(
                        dataset.iloc[rIndex, :].values.reshape(1, -1), n_neighbors=10)
                    similarities = 1-distances.flatten()
                    print(distances, indices, similarities)
                    mean_rating = dataset.iloc[rIndex, :].mean()
                    sum_wt = np.sum(similarities)-1
                    for label in knownRatingLabels[str(rIndex)]:
                        prediction = 0
                        wtd_sum = 0
                        product = 1
                        for i in range(0, len(indices.flatten())):
                            if indices.flatten()[i] == rIndex:
                                continue
                            else:
                                ratings_diff = dataset.at[indices.flatten()[
                                    i], label]
                                -np.mean(dataset.iloc[indices.flatten()[i], :])
                                product = ratings_diff * (similarities[i])
                                wtd_sum = wtd_sum + product
                        prediction = round((mean_rating + (wtd_sum/sum_wt)), 2)
                        dataset_test.at[rIndex, label] = prediction
                        print('\nPredicted rating for user -> joke %s: %f, Real rating -> %f' %
                              (label, prediction, dataset.at[rIndex, label]))
            dataset = dataset.iloc[0:maxIndex, :]
            dataset_test = dataset_test.iloc[0:maxIndex, :]
            frames.append(dataset)
            frames_test.append(dataset_test)
            print('finish %s' % f)

    pd.concat(frames, ignore_index=True).to_excel(writer, sheet_name='dataset')
    pd.concat(frames_test, ignore_index=True).to_excel(
        writer, sheet_name='dataset_test')
