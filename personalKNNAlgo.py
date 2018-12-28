import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[4,4]],'r':[[10,9],[6,5],[11,8]]}
newFeatures = [2,4]

# for i in dataset:
#     for j in dataset[i]:
#         plt.scatter(j[0],j[1],s =100, color = i)
#
# plt.scatter(newFeatures[0],newFeatures[1])
# plt.show()

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

result = kNearestNeightbors(dataset,newFeatures,k=3)
print(result)

for i in dataset:
    for j in dataset[i]:
        plt.scatter(j[0],j[1],s =100, color = i)

plt.scatter(newFeatures[0],newFeatures[1], color  = result)
plt.show()
