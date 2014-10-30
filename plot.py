import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition


def pca_plot(X, y, title="Principal Component Analysis", save=None):
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    variability = pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_)
    transformed = pca.transform(X)
    Xs = transformed[:,0]
    Ys = transformed[:,1]
    y = numpy.array(y.T.tolist()[0])
    COLOR = (y-y.min())/y.max()
    # cm = plt.get_cmap("HOT")
    plt.scatter(Xs, Ys, c=COLOR, s=15, marker='o', edgecolors='none')
    plt.title(title + "\n%s %s" % variability)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
    plt.clf()


def plot_and_save_all_pca(features, property_sets):
    for feat_name, feat in features.items():
        for prop_name, prop in property_sets:
            title = feat_name + " " + prop_name
            save = feat_name + "_" + prop_name + ".png"
            pca_plot(feat, prop, title=title, save=save)


def pca_plot_3d(X, y, title="Principal Component Analysis"):
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    print pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_)
    transformed = pca.transform(X)
    Xs = transformed[:,0]
    Ys = transformed[:,1]
    Zs = transformed[:,2]
    y = numpy.array(y.T.tolist()[0])
    COLOR = (y-y.min())/y.max()
    cm = plt.get_cmap("HOT")
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xs, Ys, Zs, c=COLOR,s=80, marker='o', edgecolors='none')
    plt.title(title)
    plt.show()
    plt.clf()


