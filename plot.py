import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import tanh
from sklearn import decomposition
from pybrain.tools.functions import sigmoid



def pca_plot(X, y, title="Principal Component Analysis", save=None, segment=False):
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    variability = pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_)
    transformed = pca.transform(X)
    Xs = transformed[:,0]
    Ys = transformed[:,1]

    if len(y.shape) > 1 and y.shape[1] == 3:
        red, green, blue = y.T
        red = numpy.matrix((red - red.min()) / (red.max() - red.min()))
        green = numpy.matrix((green - green.min()) / (green.max() - green.min()))
        blue = numpy.matrix((blue - blue.min()) / (blue.max() - blue.min()))
        alpha = numpy.matrix(numpy.ones(blue.shape)) * .75
        if segment:
            red = red > 0.5
            green = green > 0.5
            blue = blue > 0.5
        COLOR = numpy.concatenate((red.T, green.T, blue.T, alpha.T), 1)
        if segment:
            mm = numpy.array((red & green & blue).tolist()[0])
            COLOR[mm,:] = [0.75, 0.75, 0.75, 0.75]
    else:
        COLOR = (y - y.min()) / (y.max() - y.min())
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


def plot_neural_net(X, y, clf, segment=False):
    values = X
    pca_plot(X, y, save="00_conn.png", segment=segment)
    counter = 1
    for i, layer in enumerate(clf.nn.modulesSorted):
        name = layer.__class__.__name__
        if name == "BiasUnit":
            continue

        try:
            conn = clf.nn.connections[layer][0]
        except IndexError:
            continue

        if "Linear" not in name:
            if "Sigmoid" in name:
                add = "sigmoid"
                values = sigmoid(values)
            elif "Tanh" in name:
                add = "tanh"
                values = tanh(values)
            pca_plot(values, y, save="%02d_conn_%s.png" % (counter, add), segment=segment)
            counter += 1
        shape = (conn.outdim, conn.indim)
        temp = numpy.dot(numpy.reshape(conn.params, shape), values.T)
        pca_plot(temp.T, y, save="%02d_conn.png" % counter, segment=segment)
        counter += 1
        values = temp.T


def plot_feature_errors(values, property_name):
    fig, ax = plt.subplots()

    errors = {}
    for feature, methods in values.items():
        for method, error in methods.items():
            try:
                errors[method].append(error)
            except:
                errors[method] = [error]

    width = 0.9 / len(errors)

    colors = ['r', 'y', 'g', 'c', 'b', 'm', 'k']
    ind = numpy.arange(len(values.keys()))
    for i, (feature, errors) in enumerate(errors.items()):
            plt.bar(ind+i*width, errors, width, color=colors[i], label=feature)

    plt.title(property_name)
    plt.xticks(rotation=10)
    ax.set_xticks(ind + (len(errors)/2 * width))
    ax.set_xticklabels(values.keys())
    ax.set_ylabel("Mean Absolute Error (eV)")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()


def plot_method_errors(values, property_name):
    fig, ax = plt.subplots()
    
    errors = {}
    for feature, methods in values.items():
        for method, error in methods.items():
            try:
                errors[feature].append(error)
            except:
                errors[feature] = [error]
    methods = methods.keys()
    width = 0.9 / len(errors)

    colors = ['r', 'y', 'g', 'c', 'b', 'm', 'k']
    ind = numpy.arange(len(methods))
    for i, (method, errors) in enumerate(errors.items()):
        plt.bar(ind+i*width, errors, width, color=colors[i], label=method)

    plt.title(property_name)
    plt.xticks(rotation=10)
    ax.set_xticks(ind + (len(errors)/2 * width))
    ax.set_xticklabels(methods)
    ax.set_ylabel("Mean Absolute Error (eV)")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()