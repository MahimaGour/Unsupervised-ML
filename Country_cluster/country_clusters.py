import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import plotly.express as exp

df = pd.read_csv('Country_data.csv')
df.head()

labels = pd.read_csv('data_dic.csv')
labels.head(10)

df.corr()

fig = plt.figure(figsize=(10,10))
ax = sb.heatmap(df.corr(), annot=True)
plt.show()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.drop('country', axis=1))

scaled_df = pd.DataFrame(data=scaled_data, columns=df.columns[1:])
scaled_df['country'] = df['country']
scaled_df.head()

exp.histogram(data_frame=df, x='gdpp',nbins=167, opacity=0.75, barmode='overlay')

exp.scatter(data_frame=df, x='child_mort', y='health', color='country')

data = scaled_df.drop('country', axis=1)

ssd = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
  kmeans.fit(data)
  ssd.append(kmeans.inertia_)

plt.plot(range(1,11), ssd, '*-')
plt.title('The Elbow Method')
plt.xlabel('No. of clusters')
plt.ylabel('SSD (sum of squared distances)')
plt.show()

kmeans= KMeans(n_clusters=3, init='k-means++', random_state=0)
kmeans.fit(data)
pred = kmeans.labels_

exp.scatter(data_frame= df,x = 'gdpp', y='income', color=kmeans.labels_)

pca = PCA(n_components=2)
pca_model = pca.fit_transform(data)
data_transform = pd.DataFrame(data = pca_model, columns = ['PCA1', 'PCA2'])
data_transform['Cluster'] = pred

data_transform.head()

plt.figure(figsize=(8,8))
g = sb.scatterplot(data=data_transform, x='PCA1', y='PCA2', palette=sb.color_palette(n_colors=3), hue='Cluster')
title = plt.title('Countries Clusters with PCA')