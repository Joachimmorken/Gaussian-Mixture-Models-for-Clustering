from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import itertools

from scipy import linalg


import numpy as np
import pandas as pd
import matplotlib as mpl

import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

colors = ["g.", "r.", "c."]


def load_dataset():
    filename = "seeds_dataset.txt"
    dataset = np.loadtxt(filename)

    #Scaling the data
    x = StandardScaler().fit_transform(dataset)

    #Projection to 2D
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data= principalComponents, columns= ["Component 1", "Component 2"])
    dataset = principalDf.values
    return dataset



def kmeans(X, title):
    X = load_dataset()
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    #Visualization of the clustering
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    for i in range (0, len(X)):
        plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
    plt.title(title)
    plt.show()


def gaussMix():
    X = load_dataset()
    gauss = GaussianMixture(n_components=3)
    gauss.fit(X)



    #http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html
def gaussMix(X, Y_, means, covariances, index, title):
    Y = GaussianMixture(n_components=3, covariance_type="full").fit(X)
    color_iter = itertools.cycle(['navy', 'cornflowerblue', 'gold',
                                  'darkorange'])
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.show()

    
X = load_dataset()

kmeans(X, "KMeans")

gmm = GaussianMixture(n_components=3, covariance_type="full").fit(X)
gaussMix(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture')


