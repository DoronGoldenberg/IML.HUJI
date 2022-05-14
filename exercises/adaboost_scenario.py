import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default = "browser"


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    module = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    loss_train, loss_test = list(), list()
    for t in range(1, n_learners + 1):
        loss_train.append(module.partial_loss(train_X, train_y, t))
        loss_test.append(module.partial_loss(test_X, test_y, t))

    go.Figure([go.Scatter(x=np.arange(1, n_learners + 1), y=loss_train, mode='markers+lines', name=r'$Train Loss$'),
               go.Scatter(x=np.arange(1, n_learners + 1), y=loss_test, mode='markers+lines', name=r'$Test Loss$')],
              layout=go.Layout(title=r"$\text{AdaBoost Error As Function Of Fitted Learners}$",
                               xaxis_title="r$\\text{Fitted Learners}$",
                               yaxis_title="r$\\text{misclassification error}$")).show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    colors = np.array(test_y == 1, dtype=np.int_)
    symbols = np.array(["circle", "x"])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{t} Iterations}}$" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(lambda _X: module.partial_predict(_X, t), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=colors, symbol=symbols[colors],
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"$\textbf{{Decision Boundaries Of Model}}$", margin=dict(t=100))\
        .update_xaxes(visible=False).update_yaxes(visible=False).show()

    # Question 3: Decision surface of best performing ensemble
    from IMLearn.metrics.loss_functions import accuracy
    best_iteration = np.argmin(loss_test) + 1
    acc = accuracy(test_y, module.partial_predict(test_X, best_iteration))
    go.Figure([decision_surface(lambda _X: module.partial_predict(_X, best_iteration), lims[0], lims[1], showscale=False),
               go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                          marker=dict(color=colors,
                                      symbol=symbols[colors],
                                      colorscale=[custom[0], custom[-1]],
                                      line=dict(color="black", width=1)))],
              layout=go.Layout(title=rf"$\textbf{{Decision Boundaries Of Model Of Size {best_iteration} (Accuracy = {acc})}}$")).show()

    # Question 4: Decision surface with weighted samples
    size_ = 5 * module.D_ / np.max(module.D_)
    colors = np.array(train_y == 1, dtype=np.int_)
    go.Figure([decision_surface(module.predict, lims[0], lims[1], showscale=False),
               go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                          marker=dict(color=colors,
                                      symbol=symbols[colors],
                                      size=size_,
                                      colorscale=[custom[0], custom[-1]],
                                      line=dict(color="black", width=1)))],
              layout=go.Layout(title=rf"$\textbf{{Training Set Weight}}$")).show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
