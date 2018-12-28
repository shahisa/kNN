import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random


def kNearestNeightbors(data,predict,k=3):
    if len(data) >= k:
        warnings.warn(' K is set to a value less than total voting groups')
    distances = []
    for group in data:
        for features in data[group]:
            euclideanDistance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclideanDistance,group])
    votes = [i[1] for i in sorted(distances) [:k]]
    voteResults = Counter(votes).most_common(1)[0][0]
    return voteResults

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace =True)
fullData = df.astype(float).values.tolist()
random.shuffle(fullData)

testSize = 0.2
trainSet = {2:[],4:[]}
testSet = {2:[],4:[]}
trainData = fullData[:-int(testSize*len(fullData))]
testData = fullData[-int(testSize*len(fullData)):]

for i in trainData:
    trainSet[i[-1]].append(i[:-1])

for i in testData:
    testSet[i[-1]].append(i[:-1])

correct = 0
total = 0

#k =5 because scikit-learn says they are using a defualt k of 5
for group in testSet:
    for data in testSet[group]:
        vote = kNearestNeightbors(trainSet, data, k=5)
        if group == vote:
            correct += 1
        total +=1
print('Accuracy:', correct/total)
