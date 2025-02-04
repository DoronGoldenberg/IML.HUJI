import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go

from sklearn.metrics import roc_curve


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values_record = list()
    weights_record = list()

    def callback(weights, val, **kwargs):
        values_record.append(val)
        weights_record.append(weights)

    return callback, values_record, weights_record


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for module, name in [(L1, "L1"), (L2, "L2")]:
        norms_plot = list()
        for eta in etas:
            lr = FixedLR(eta)
            callback, vals, weights = get_gd_state_recorder_callback()
            gd = GradientDescent(learning_rate=lr, tol=0, callback=callback)
            f = module(init)
            gd.fit(f, None, None)
            best_loss = np.array(vals).min()
            plot_descent_path(module, np.array([init, *weights]), 'module {}: eta = {}, best_loss = {}'.format(name, eta, best_loss)).show()
            norms_plot.append(go.Scatter(x=np.arange(0, len(vals)), y=np.array(vals), mode='lines',
                                         name=rf'$Module {name}: eta = {eta} (best_loss = {best_loss})$'))
        go.Figure(norms_plot, layout=go.Layout(title=rf"$Convergence rate for module {name} with Fixed LR$",
                                               xaxis_title="r$\\text{Iteration}$",
                                               yaxis_title="r$\\text{Norm}$")).show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    norms_plot = list()
    for gamma in gammas:
        lr = ExponentialLR(eta, gamma)
        callback, vals, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=lr, tol=0, callback=callback)
        f = L1(init)
        gd.fit(f, None, None)
        best_loss = np.array(vals).min()
        norms_plot.append(go.Scatter(x=np.arange(0, len(vals)), y=np.array(vals), mode='lines',
                                     name=rf'$Module L1: eta = {eta}, gamma = {gamma}, (best_loss = {best_loss})$'))

    # Plot algorithm's convergence for the different values of gamma
    go.Figure(norms_plot, layout=go.Layout(title=rf"$Convergence rate for module L1 with Exponential LR$",
                                           xaxis_title="r$\\text{Iteration}$",
                                           yaxis_title="r$\\text{Norm}$")).show()

    # Plot descent path for gamma=0.95
    gamma = 0.95
    for module, name in [(L1, "L1"), (L2, "L2")]:
        lr = ExponentialLR(eta, gamma)
        callback, vals, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=lr, tol=0, callback=callback)
        f = module(init)
        gd.fit(f, None, None)
        best_loss = np.array(vals).min()
        plot_descent_path(module, np.array([init, *weights]),
                          'module {} with Exponential LR: eta = {}, gamma = {}, best_loss = {}'.format(name, eta, gamma, best_loss)).show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    # Plotting convergence rate of logistic regression over SA heart disease data
    module = LogisticRegression().fit(X_train, y_train)
    fpr, tpr, thresholds = roc_curve(y_test, module.predict_proba(X_test))
    opt = thresholds[np.argmax(tpr - fpr)]
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - alfa*}}={opt:.6f} \text{{(Test error={module.loss(X_test, y_test):.6f})}}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    K = 5
    indexes = np.arange(X_train.shape[0])
    np.random.shuffle(indexes)
    validations = np.array([[i in v for i in indexes] for v in np.array_split(indexes, K)])
    for p in ["l1", "l2"]:
        opt_lam = 0.
        opt_score = None
        for lam in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
            score = 0.
            for val in validations:
                module = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                                            penalty=p, lam=lam).fit(X_train[indexes[~val]], y_train[indexes[~val]])
                score += module.loss(X_train[indexes[val]], y_train[indexes[val]])
            if opt_score is None or score < opt_score:
                opt_score = score
                opt_lam = lam

        module = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                                    penalty=p, lam=opt_lam).fit(X_train, y_train)
        print('Module {} - opt_lambda = {}, (test error = {})'.format(p, opt_lam, module.loss(X_test, y_test)))


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
