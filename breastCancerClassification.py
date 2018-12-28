import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999, inplace = True)
df.drop(['id'],1,inplace=True)
# x is for the features and y is for the labels
X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.3)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)
#accuray is fluxuating because of lack of data points / ordering / no normulization
# range of change is 1-5% fluxuation with 95% accuracy being the lowest amount

#below is for one prediction
# exampleMeasures = np.array([4,3,2,2,1,2,8,2,1])
#for two predictions
exampleMeasures = np.array([[4,3,2,2,1,2,8,2,1],[4,3,2,2,1,2,8,2,1]])
#for the number of prediction changes you have to change the first value in reshape(#,-1)
exampleMeasures = exampleMeasures.reshape(len(exampleMeasures),-1)
prediction = clf.predict(exampleMeasures)

print(prediction)
