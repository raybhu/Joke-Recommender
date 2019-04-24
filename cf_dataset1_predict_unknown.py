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
# jokes = []
# for f in os.listdir(JOKE_PATH):
#     if 'init' in f:
#         soup = BeautifulSoup(open(JOKE_PATH + '/' + f), 'lxml')
#         jokes.append(soup.body.text.strip())
# with open(os.path.abspath(
#         os.path.dirname(__file__)+'/jokes.json'), 'w', encoding='utf-8') as f:
#     f.write(json.dumps(jokes))
#     f.close()

dataset_path = DATASETS_PATH['dataset1']
with pd.ExcelWriter('dataset.xlsx') as writer:
    for f in os.listdir(dataset_path):
        if 'jester' in f:
            dataset = pd.read_excel(
                dataset_path+'/'+f, header=None, usecols='B:CW')
            predictedJokesLabels = {}
            for rIndex, row in dataset.iterrows():
                # print(row)
                for clabel, value in row.iteritems():
                    if int(value) == 99:
                        dataset.at[rIndex, clabel] = 0
                        if str(rIndex) not in predictedJokesLabels:
                            predictedJokesLabels[str(rIndex)] = []
                        predictedJokesLabels[str(rIndex)].append(clabel)
            model_knn = NearestNeighbors(metric='cosine', algorithm='auto')
            model_knn.fit(dataset)
            maxIndex = 10
            for rIndex, row in dataset.iterrows():
                if int(rIndex) == maxIndex:
                    break
                if str(rIndex) in predictedJokesLabels:
                    distances, indices = model_knn.kneighbors(
                        dataset.iloc[rIndex, :].values.reshape(1, -1), n_neighbors=10)
                    similarities = 1-distances.flatten()
                    print(distances, indices, similarities)
                    mean_rating = dataset.iloc[rIndex, :].mean()
                    sum_wt = np.sum(similarities)-1
                    for label in predictedJokesLabels[str(rIndex)]:
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
                        dataset.at[rIndex, label] = prediction
                        print('\nPredicted rating for user -> joke %s: %f' %
                              (label, prediction))
            dataset = dataset.iloc[0:maxIndex, :]
            dataset.to_excel(writer, sheet_name=f)
