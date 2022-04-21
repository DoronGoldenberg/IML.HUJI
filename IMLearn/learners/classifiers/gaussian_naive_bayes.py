from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, inverse, n = np.unique(y, return_inverse=True, return_counts=True)
        encoded = np.zeros((self.classes_.size, inverse.size))
        encoded[inverse, np.arange(inverse.size)] = 1
        self.mu_ = ((encoded @ X).T / n).T
        self.vars_ = ((encoded @ ((X - self.mu_[inverse]) * (X - self.mu_[inverse]))).T / n).T
        self.pi_ = n / y.size

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        a = - 1. / (2. * self.vars_.T)
        b = (self.mu_ / self.vars_).T
        c = np.log(self.pi_) - np.sum(np.log(self.vars_), axis=1) / 2 - np.sum(self.mu_ * self.mu_ / (2. * self.vars_), axis=1)
        return self.classes_[np.argmax((X * X) @ a + X @ b + c, axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        a = - 1. / (2. * self.vars_.T)
        b = (self.mu_ / self.vars_).T
        c = - np.sum(self.mu_ * self.mu_ / (2. * self.vars_), axis=1)

        likelihood = np.exp((X * X) @ a + X @ b + c) * self.pi_ / np.sqrt(np.prod(self.vars_, axis=1))
        likelihood = (likelihood.T / np.sum(likelihood, axis=1)).T
        return likelihood

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
