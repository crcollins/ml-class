import numpy

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from scipy import tanh
import scipy.spatial as spatial

from sklearn import decomposition
from sklearn.manifold import TSNE

from pybrain.tools.functions import sigmoid


def fmt(x, y, name):
    return name


class FollowDotCursor(object):
    """
    Display the x,y location of the nearest data point.

    Adapted from http://stackoverflow.com/questions/13306519/get-data-from-plot-with-matplotlib
    """
    def __init__(self, ax, x, y, names, tolerance=5, formatter=fmt, offsets=(-20, 20)):
        try:
            x = numpy.asarray(x, dtype='float')
        except (TypeError, ValueError):
            x = numpy.asarray(mdates.date2num(x), dtype='float')
        y = numpy.asarray(y, dtype='float')
        self._points = numpy.column_stack((x, y))
        self._names = names
        self.offsets = offsets
        self.scale = x.ptp()
        self.scale = y.ptp() / self.scale if self.scale else 1
        self.tree = spatial.cKDTree(self.scaled(self._points))
        self.formatter = formatter
        self.tolerance = tolerance
        self.ax = ax
        self.fig = ax.figure
        self.ax.xaxis.set_label_position('top')
        self.dot = ax.scatter(
            [x.min()], [y.min()], s=130, color='green', alpha=0.7)
        self.annotation = self.setup_annotation()
        plt.connect('motion_notify_event', self)

    def scaled(self, points):
        points = numpy.asarray(points)
        return points * (self.scale, 1)

    def __call__(self, event):
        ax = self.ax
        # event.inaxes is always the current axis. If you use twinx, ax could be
        # a different axis.
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
        elif event.inaxes is None:
            return
        else:
            inv = ax.transData.inverted()
            x, y = inv.transform([(event.x, event.y)]).ravel()
        annotation = self.annotation
        x, y, name = self.snap(x, y)
        annotation.xy = x, y
        annotation.set_text(self.formatter(x, y, name))
        self.dot.set_offsets((x, y))
        bbox = ax.viewLim
        event.canvas.draw()

    def setup_annotation(self):
        """Draw and hide the annotation box."""
        annotation = self.ax.annotate(
            '', xy=(0, 0), ha = 'right',
            xytext = self.offsets, textcoords = 'offset points', va = 'bottom',
            bbox = dict(
                boxstyle='round,pad=0.5', fc='yellow', alpha=0.75),
            arrowprops = dict(
                arrowstyle='->', connectionstyle='arc3,rad=0'))
        return annotation

    def snap(self, x, y):
        """Return the value in self.tree closest to x, y."""
        dist, idx = self.tree.query(self.scaled((x, y)), k=1, p=1)
        try:
            x, y = self._points[idx]
            return x, y, self._names[idx]
        except IndexError:
            # IndexError: index out of bounds
            x, y = self._points[0]
            return x, y, '---'


def pca_plot(X, y, title="Principal Component Analysis", save=None, segment=False, names=None, inspect=False, perturb=None):
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
        COLOR = numpy.squeeze(numpy.array(COLOR))

    if perturb is not None:
        rx = numpy.random.randn(*Xs.shape)
        ry = numpy.random.randn(*Ys.shape)
        Xs += rx * perturb
        Ys += ry * perturb
    plt.scatter(Xs, Ys, c=COLOR, s=15, marker='o', edgecolors='none')
    plt.title(title + "\n%s %s" % variability)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    if save is None or inspect:
        if inspect:
            ax = plt.gca()
            cursor = FollowDotCursor(ax, Xs, Ys, names)
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

    method_ordering = [
        "Mean",
        "Linear",
        "LinearRidge",
        "SVM (Gaussian)",
        "SVM (Laplacian)",
        "k-NN",
        "Tree",
        "AdaBoost",
        "Gradient Boost",
    ]

    errors = {}
    for feature, methods in values.items():
        for method in method_ordering:
            error = methods[method]
            try:
                errors[feature].append(error)
            except:
                errors[feature] = [error]
    width = 0.9 / len(errors)

    colors = [
        "#F44336",
        "#9C27B0",
        "#3F51B5",
        "#03A9F4",
        "#4CAF50",
        "#CDDC39",
        "#FFC107",
        "#E91E63",
        "#673AB7",
        "#2196F3",
        "#00BCD4",
        "#8BC34A",
        "#FFEB3B",
        "#FF5722",
    ]
    vector_ordering = [
        "null_feature",
        "binary_feature",
        "flip_binary_feature",
        "decay_feature",
        "centered_decay_feature",
        "signed_centered_decay_feature",
        "gauss_decay_feature",
        "coulomb_feature",
        "pca_coulomb_feature",
    ]
    ind = numpy.arange(len(method_ordering))
    for i, method in enumerate(vector_ordering):
        values = errors[method]
        means, stds = zip(*values)
        plt.bar(ind+i*width, means, width, yerr=stds, color=colors[i], ecolor='k', label=method)

    plt.title(property_name, fontsize=24)
    plt.xticks(rotation=10)
    ax.set_xticks(ind + (len(values)/2 * width))
    ax.set_xticklabels(method_ordering, fontsize=18)
    ax.set_ylabel("Mean Absolute Error (eV)", fontsize=18)
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()


def plot_surface(x, y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = numpy.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
    plt.show()


def plot_multi_surface(xs, ys, Zs):
    fig = plt.figure()
    for x, y, Z in zip(xs, ys, Zs):
        ax = fig.gca(projection='3d')
        X, Y = numpy.meshgrid(x, y)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.5, cmap=cm.coolwarm, linewidth=0)
    plt.show()


def plot_TSNE(X, y, names=None):
    model = TSNE(n_components=2, random_state=0)
    start = time.time()

    a = numpy.arange(X.shape[0])
    rand = numpy.random.choice(a, 3200)

    res = model.fit_transform(X[rand,:])

    bla = []
    for prop in (y, y, y):
        y = prop[rand]
        colors = (y - y.min()) / (y.max() - y.min())
        bla.append(colors.T.tolist()[0])

    temp = numpy.array(bla)
    plt.scatter(res[:,0], res[:,1], color=temp.T)
    print time.time() - start

    ax = plt.gca()
    cursor = FollowDotCursor(ax, res[:,0], res[:,1], names)
    # plt.show()


