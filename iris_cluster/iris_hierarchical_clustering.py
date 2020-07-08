"""
# Objective:
1. Plotting dendogram to find optimal no. of clusters
2. Use Hierarchical/ agglomerative clustering approach
"""

#importing necessary libraries/api
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
# %config Inlinebackend.figure_format = 'retina'
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

#loading dataset
iris = datasets.load_iris()

#making dataframe
x_pd = pd.DataFrame(iris.data, columns=iris.feature_names)
x_pd.head()

#plotting dendogram
dendogram = sch.dendrogram(sch.linkage(x_pd, method='ward'))
plt.title('Dendogram')
plt.xlabel('x_pd')
plt.ylabel('Distances')
plt.show()

#making model
hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
cluster = hc.fit_predict(x_pd)

#adding column to dataframe
x_pd['cluster'] = cluster
x_pd.cluster.value_counts()

#plotting predicted clusters
sb.scatterplot(data=x_pd, x='sepal length (cm)', y='sepal width (cm)',  palette=sb.color_palette(n_colors=4), hue='cluster')
title = plt.title(' Clusters')
plt.show()