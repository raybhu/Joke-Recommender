import os
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, Comment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

JOKE_PATHS = os.path.abspath(
    os.path.dirname(__file__)+'/Dataset/jokes/')
DATASETS_PATH = {'dataset1': os.path.abspath(
    os.path.dirname(__file__)+'/Dataset/jester_dataset_1/'), 'dataset2': os.path.abspath(
    os.path.dirname(__file__)+'/Dataset/jester_dataset_2/'), 'dataset3': os.path.abspath(
    os.path.dirname(__file__)+'/Dataset/jester_dataset_3/')}
jokes = []
for f in os.listdir(JOKE_PATHS):
    if 'init' in f:
        soup = BeautifulSoup(open(JOKE_PATHS + '/' + f), 'lxml')
        jokes.append(soup.body.text.strip())
with open(os.path.abspath(
        os.path.dirname(__file__)+'/jokes.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(jokes))
    f.close()

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(jokes)
# print(type(X), X.toarray())
dataset_path = DATASETS_PATH['dataset1']
with pd.ExcelWriter('CB_ResultofBinaryPrediction.xlsx') as writer:
    for f in os.listdir(dataset_path):
        if f != 'jester-data-1.xlsx':
            continue
        if 'jester' in f:
            dataset = pd.read_excel(
                dataset_path+'/'+f, header=None, usecols='B:CW')
            dataset_test = np.zeros((dataset.shape[0], 100))
            knownRatingLabels = {}
            maxIndex = 10
            for rIndex, row in dataset.iterrows():
                if maxIndex and int(rIndex) == maxIndex:
                    break
                for clabel, value in row.iteritems():
                    if float(value) == 99:
                        dataset.at[rIndex, clabel] = 0
                    elif float(value) < 0:
                        dataset.at[rIndex, clabel] = -1
                        cos = cosine_similarity(X.getrow(int(clabel)-1), X)
                        cos[0, int(clabel)-1] = 0
                        maxSimilarityJoke = np.argmax(cos, axis=1)+1
                        print(cos, maxSimilarityJoke)
                        # dataset_test.at(rIndex, maxSimilarityJoke) = -1
                        dataset_test[rIndex, np.argmax(cos, axis=1)] = -1
                    elif float(value) > 0:
                        dataset.at[rIndex, clabel] = 1
                        cos = cosine_similarity(X.getrow(int(clabel)-1), X)
                        cos[0, int(clabel)-1] = 0
                        maxSimilarityJoke = np.argmax(cos, axis=1)+1
                        # dataset_test.at(rIndex, maxSimilarityJoke) = 1
                        dataset_test[rIndex, np.argmax(cos, axis=1)] = 1
                        print(cos, maxSimilarityJoke, dataset_test)

            dataset.iloc[0:maxIndex, :].to_excel(
                writer, sheet_name='dataset')
            pd.DataFrame(dataset_test, columns=range(1, 101)).iloc[0:maxIndex, :].to_excel(
                writer, sheet_name='dataset_test')
           
