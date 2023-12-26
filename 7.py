from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sklearn.metrics as metrics
import pandas as pd
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

plt.figure(figsize=(14, 7))
colormap = np.array(['red', 'lime', 'black'])

# REAL PLOT
plt.subplot(1, 3, 1)
plt.title('Real')
plt.scatter(X[:, 2], X[:, 3], c=colormap[y])

# K-PLOT
model = KMeans(n_clusters=3, random_state=0).fit(X)
plt.subplot(1, 3, 2)
plt.title('KMeans')
plt.scatter(X[:, 2], X[:, 3], c=colormap[model.labels_])

print('The accuracy score of K-Mean: ', metrics.accuracy_score(y, model.labels_))
print('The Confusion matrix of K-Mean:\n', metrics.confusion_matrix(y, model.labels_))

# GMM PLOT
gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
y_cluster_gmm = gmm.predict(X)
plt.subplot(1, 3, 3)
plt.title('GMM Classification')
plt.scatter(X[:, 2], X[:, 3], c=colormap[y_cluster_gmm])
plt.show()

print('The accuracy score of EM: ', metrics.accuracy_score(y, y_cluster_gmm))
print('The Confusion matrix of EM:\n', metrics.confusion_matrix(y, y_cluster_gmm))
