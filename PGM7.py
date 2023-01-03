import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
import pandas as pd
import numpy as np
import sklearn.metrics as sm

l1 = [0,1,2]

def rename(s):
	l2 = []
	for i in s:
		if i not in l2:
			l2.append(i)

	for i in range(len(s)):
		pos = l2.index(s[i])
		s[i] = l1[pos]
	return s

iris = datasets.load_iris()

X = pd.DataFrame(iris.data)
X.columns =['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y = pd.DataFrame(iris.target)
y.columns = ['Targets']

plt.figure(figsize=(14,14))
colormap = np.array(['red', 'lime', 'black'])

plt.subplot(1, 3, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets],s=40)
plt.title('Real Clusters')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

model = KMeans(n_clusters=3)
model.fit(X) 

plt.subplot(1, 3, 2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_],s=40)
plt.title('K-Means Clustering')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

km = rename(model.labels_)
print("Accuracy of KMeans is ",sm.accuracy_score(y, km))

scaler = preprocessing.StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
xs = pd.DataFrame(xsa, columns = X.columns)

gmm = GaussianMixture(n_components=3)
gmm.fit(xs)
gmm_y = gmm.predict(xs)

plt.subplot(1, 3, 3)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[gmm_y], s=40)
plt.title('GMM Clustering')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

em = rename(gmm_y)
print("Accuracy of EM is ",sm.accuracy_score(y, em))
print('Observation: The GMM using EM algorithm based clustering matched the true labels more closely than the Kmeans')