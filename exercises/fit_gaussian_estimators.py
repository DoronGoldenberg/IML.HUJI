from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    module = UnivariateGaussian()
    mu = 10
    var = 1

    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(mu, var, 1000)
    module.fit(X)
    print("Estimation mu: {}, Estimation var: {}".format(module.mu_, module.var_))  # print 9.954743292509804, 0.9752096659781323

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.arange(10, 1010, 10)
    estimated_mean = np.array([module.fit(X[:n]).mu_ for n in ms])
    absolute_error = np.abs(estimated_mean - mu)
    go.Figure([go.Scatter(x=ms, y=absolute_error, mode='markers+lines', name=r'$|\widehat\mu-\mu|$'),
               go.Scatter(x=ms, y=np.zeros_like(ms), mode='lines', name=r'$\mu$')],
              layout=go.Layout(
                  title=r"$\text{(2) Absolute Error of Expectation's Estimation As Function Of Number Of Samples}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="r$|error|$")).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    go.Figure([go.Scatter(x=X, y=module.pdf(X), mode='markers', name=r'$pdf$')],
              layout=go.Layout(
                  title=r"$\text{(3) Probability Density As Function Of The Sampled Values}$",
                  xaxis_title="$\\text{samples value}$",
                  yaxis_title="$\\text{Probability density}$")).show()


def test_multivariate_gaussian():
    module = MultivariateGaussian()
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1.0, 0.2, 0.0, 0.5],
                    [0.2, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.5, 0.0, 0.0, 1.0]])

    # Question 4 - Draw samples and print fitted model
    X = np.random.multivariate_normal(mu, cov, 1000)
    module.fit(X)
    print("Estimation mu:")
    print(module.mu_)  # print [-0.02282878 -0.04313959  3.9932571  -0.02038981]
    print("Estimation cov:")
    print(module.cov_)
    # print [[ 0.91667608  0.16634444 -0.03027563  0.46288271]
    #        [ 0.16634444  1.9741828  -0.00587789  0.04557631]
    #        [-0.03027563 -0.00587789  0.97960271 -0.02036686]
    #        [ 0.46288271  0.04557631 -0.02036686  0.9725373 ]]

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    values = np.array([[module.log_likelihood(np.array([x1, 0, x3, 0]), cov, X) for x1 in f1] for x3 in f3])
    go.Figure(data=go.Heatmap(x=f1, y=f3, z=values, hoverongaps=False, name=r'$log likelihood$'),
              layout=go.Layout(
                  title=r"$\text{(5) Log Likelihood Of The Samples As Function Of Expectation [f1, 0, f3, 0]}$",
                  xaxis_title="$\\text{f1}$",
                  yaxis_title="$\\text{f3}$")).show()

    # Question 6 - Maximum likelihood
    i, j = np.unravel_index(np.argmax(values),values.shape)
    print("Maximum likelihood([f1, 0, f3, 0]) = [{:.3f} 0 {:.3f} 0]".format(f1[j], f3[i]))  # print [-0.05, 0, 3.970, 0]


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
