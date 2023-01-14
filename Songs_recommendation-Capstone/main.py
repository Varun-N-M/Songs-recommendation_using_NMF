
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import NMF

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('spotify.xls',index_col=0)

print(df.head().to_string())
print(df.info())
print(df.describe().to_string())

x = df.values
nmf = NMF(n_components=100,max_iter=1000,random_state=100)

nmf.fit(x)
user_matrix = nmf.transform(x)
song_matrix = nmf.components_.T

print(user_matrix)
print(song_matrix)
print(user_matrix.shape)
print(song_matrix.shape)

def col_dist(u1,u2):
    return pow(sum(pow(u1[x]-u2[x],2) for x in range(len(u1))),0.5)
print(col_dist(user_matrix[0],user_matrix[1]))

def dist_from_remaining_users(b_user,user_matrix):
    dist = []
    for i in range(len(user_matrix)):
        if b_user != i:
            dist.append(col_dist(user_matrix[b_user],user_matrix[i]))
    return dist

d1 = dist_from_remaining_users(0,user_matrix)
print(len(d1))

nearest_users = np.argsort(d1)[:5]
print(nearest_users)

for i in nearest_users:
    print('Songs heard by',i,'are:')
    temp = df.iloc[i]
    print(temp[temp.values!=0].index)

def top_n_songs(nearest_users,df):
    temp = df.iloc[nearest_users]
    dict1 = temp.max().to_dict()

    sorted_dict = sorted(dict1.items(),key=lambda keyvalue:(keyvalue[1],keyvalue[0]),reverse=True)[:10]
    return [x[0] for x in sorted_dict]

print(top_n_songs(nearest_users,df))

wcss = {}
for k in range(1,51):
    kmeans = KMeans(n_clusters=k,max_iter=1000,random_state=42)
    kmeans.fit(song_matrix)
    wcss[k] = kmeans.inertia_

plt.plot(list(wcss.keys()),list(wcss.values()))
plt.xlabel('number of clusters')
plt.ylabel('within cluster sum of square')
plt.show()

def recommended_songs(n_clusters,df,song_matrix,song_name,n_recommendation):

    kmeans = KMeans(n_clusters=n_clusters,max_iter=1000,random_state=42).fit(song_matrix)
    index_in_songs = df.columns.to_list().index(song_name)
    song_vector = song_matrix[index_in_songs]
    all_songs_in_cluster = kmeans.predict(song_matrix)

    songs_in_selected_cluster = [x for x in range(len(all_songs_in_cluster))if all_songs_in_cluster[x]==kmeans.predict([song_vector])]
    song_cluster = song_matrix[songs_in_selected_cluster]

    neighbors = NearestNeighbors(n_neighbors=n_recommendation)
    neighbors.fit(song_cluster)

    recommended_songs = neighbors.kneighbors([song_matrix[index_in_songs]])
    songs = df.columns

    return [songs[x] for x in recommended_songs[1][0]]

print(recommended_songs(8,df,song_matrix,'song_101',10))