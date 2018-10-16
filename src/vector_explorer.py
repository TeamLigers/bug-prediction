from Py2Vec_model import Py2Vec
import os
import numpy as np
import _pickle as pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json

model = Py2Vec(os.getcwd() + "\py2vec\Lab41.json")

with open("py2vec\py2vec_model3.pkl", "rb") as f:
    embeddings = pickle.load(f)

Lab41Embeddings = json.load(open(os.getcwd() + "\py2vec\Lab41.json"))


def kmeans(data, k, num_iter):
    c = np.zeros(data.shape[0])
    centroids = np.random.rand(k, data.shape[1])

    for i in range(num_iter):
        for j in range(data.shape[0]):
            min_dist = np.sum((data[j, :] - centroids[0, :])**2)
            for l in range(k):
                dist = np.sum((data[j, :] - centroids[l, :])**2)
                if dist <= min_dist:
                    min_dist = dist
                    c[j] = l
        for j in range(k):
            clusters = np.where(c == j)
            centroids[j, :] = np.sum(data[clusters, :], axis=1) / len(clusters[0])
    return centroids, c

def get_low_dim_embs(vectors):
    tsne = TSNE(
        perplexity=30, n_components=2, init='pca', n_iter=250)
    return tsne.fit_transform(vectors)

nparray = []
for key, value in Lab41Embeddings.items():
    nparray.append(value)


nparray = np.asarray(nparray)
nparray = get_low_dim_embs(nparray)
cen, ndx = kmeans(nparray, 6, 50)
print(cen)


colors = ['r', 'g', 'b', 'y', 'm', 'c']
clusters = []
fig, ax = plt.subplots()
for i in range(6):
    clusters.append(nparray[np.where(ndx == i)[0], :])
    ax.scatter(clusters[i][:, 0], clusters[i][:, 1], s=30, color=colors[i], label='Cluster ' + str(i))
plt.show()


def plot_with_labels(low_dim_embs, labels):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(10, 10))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')
    plt.show()


def plot():
    tsne = TSNE(
        perplexity=30, n_components=2, init='pca', n_iter=250)

    plot_only = 300
    plot = []
    labels = []
    i = 0
    for key, value in embeddings.items():
        if i > plot_only:
            break
        i = i + 1
        labels.append(key)
        plot.append(value)

    plot = np.asarray(plot)
    low_dim_embs = tsne.fit_transform(plot)
    plot_with_labels(low_dim_embs, labels)


def plot2():
    tsne = TSNE(
        perplexity=30, n_components=2, init='pca', n_iter=250)

    plot = []
    labels = ['if', 'else', 'elif', 'while', 'for', 'range', 'open', 'in', 'break', 'continue', 'close', 'import', 'def', 'append', 'get']
    for word in labels:
        plot.append(Lab41Embeddings[word])

    plot = np.asarray(plot)
    low_dim_embs = tsne.fit_transform(plot)
    plot_with_labels(low_dim_embs, labels)


#plot()





# print(str(model['if']) + "\n\n\n\n\n")
#
#
# ifV = model['if']
# elifV = model['elif']
# print(np.sum((ifV - elifV)**2))
#
# print(model.closest_words('if', 5))
#
#
# # word1 is to word2 as word3 is to guess.
# word1 = 'while'
# word2 = 'for'
# word3 = 'else'
# guess = model[word3] - model[word1] + model[word2]
# print(word1 + ' is to ' + word2 + ' as ' + word3 + ' is to '
#       + str(list(f[1] for f in model.closest_words(guess, 5))))
# print(model.closest_words(guess, 5))
