import pandas
from sklearn.linear_model import LinearRegression as sklr
from sklearn.metrics import mean_squared_error

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)

    df = df.drop(index=df[df.id == 0].index)

    df = df.dropna(subset=['price'])
    df = df.drop(index=df[df.price <= 0].index).reset_index()

    df.loc[df['yr_renovated'] == 0, 'yr_renovated'] = df.loc[df['yr_renovated'] == 0, 'yr_built']
    df.loc[df['sqft_lot15'] <= 0, 'sqft_lot15'] = df.loc[df['sqft_lot15'] <= 0, 'sqft_lot']

    N = 5
    norm = (df[['lat', 'long']] - df[['lat', 'long']].min()) / (df[['lat', 'long']].max() - df[['lat', 'long']].min())
    A = np.zeros((norm.shape[0], N * N))
    bins = np.array([N, 1]) @ np.digitize(norm, np.linspace(0, 1, N-1)).T
    A[np.arange(norm.shape[0]), bins] = 1
    locations = pd.DataFrame(A, columns=['location {}'.format(i) for i in range(A.shape[1])])
    locations = locations.loc[:, (locations != 0).any(axis=0)]

    zipcode = pd.get_dummies(df.zipcode, prefix='zipcode')

    # go.Figure([go.Scatter(x=group['lat'], y=group['long'], name='location ({})'.format(zipcode), mode='markers', opacity=0.75) for zipcode, group in df.groupby(['zipcode'])]).show()

    # df = df.drop(['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15'], axis=1)
    # A.reshape((-1))[np.concatenate((np.arange(norm.shape[0]).reshape((-1, 1)), (np.digitize(norm, np.linspace(-3, 3, N-1)) @ np.array([N, 1])).reshape((-1, 1))), axis=1) @ np.array([N*N, 1])] = 1
    # print(norm)
    # go.Figure([go.Histogram2d(x=norm['lat'], y=norm['long'])]).show()

    # go.Figure([go.Scatter(x=norm['lat'], y=norm['long'], name='location', mode='markers', opacity=0.75)]).show()

    # print(*[df.sort_values(by=k)[k] for k in df.columns], sep='\n')
    # go.Figure([go.Scatter(x=df['zipcode'], y=[0]*df['zipcode'].shape[0], name='zipcode', mode='markers', opacity=0.75)]).show()
    # for k in df.columns:
    #     if k in {'id', 'date'}:
    #         continue
    #     go.Figure([go.Scatter(x=df[k], y=[0]*df[k].shape[0], name=k, mode='markers', opacity=0.75)]).show()
    # print(*['{}\t:min={},\tmax={}\n'.format(k, df.sort_values(by=k)[k].iloc[0], df.sort_values(by=k)[k].iloc[-1]) for k in df.columns])
    # func = lambda row: float(str(row['date'])[:4]) + float(str(row['date'])[:6]) % 100 / 12
    # date = df.apply(apply, axis=1)
    # print(date)
    # go.Figure([go.Scatter(x=date, y=df['price'], mode='markers')]).show()
    # print(*['{}\t: {}\n'.format(k, df.at[0, k]) for k in df.columns])
    prices = df['price']
    prices = (prices-prices.mean())/ prices.std()

    discrete = df[["bedrooms", "bathrooms", "floors", "waterfront", "view", "condition", "grade"]]
    continuous = df[["sqft_living", "sqft_lot", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "sqft_living15", "sqft_lot15"]]
    discrete = (discrete-discrete.min())/(discrete.max()-discrete.min())
    continuous = (continuous-continuous.mean())/ continuous.std()

    df = pd.concat([discrete, continuous, locations, zipcode], axis=1)
    return df, prices


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    p = dict()
    for feature in X.columns:
        cov = np.cov(X[feature], y)
        p[feature] = 0 if cov[0,1] == 0 else cov[0,1] / np.sqrt(cov[0,0] * cov[1,1])
    neg = sorted(p.items(), key=lambda item: item[1])[0]
    pos = sorted(p.items(), key=lambda item: item[1])[-1]
    neutral = sorted(p.items(), key=lambda item: np.abs(item[1]))
    print(neg, pos, neutral[0])
    # go.Figure([go.Scatter(x=X[neg[0]], y=y, name=neg[0], mode='markers', opacity=0.75)]).show()
    # go.Figure([go.Scatter(x=X[pos[0]], y=y, name=pos[0], mode='markers', opacity=0.75)]).show()
    go.Figure([go.Scatter(x=X[neutral[-1][0]], y=y, name=neutral[-1][0], mode='markers', opacity=0.75)],
              layout=go.Layout(title=r"$\text{(4) Price As Function of " + neutral[-1][0] + "} $",
                               xaxis_title=neutral[-1][0],
                               yaxis_title=r"Price")).show()
    go.Figure([go.Scatter(x=X[neutral[0][0]], y=y, name=neutral[0][0], mode='markers', opacity=0.75)],
              layout=go.Layout(title=r"$\text{(4) Price As Function of " + neutral[0][0] + "} $",
                               xaxis_title=neutral[0][0],
                               yaxis_title=r"Price")).show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X_, y_ = load_data('house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X_,y_,'')

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X_, y_, train_proportion=0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*
    module = LinearRegression()
    # m2 = sklr()

    module.fit(train_X.to_numpy(), train_y.to_numpy())
    # m2.fit(train_X.to_numpy(), train_y.to_numpy())

    # print(*['{}({}): {}, {}\n'.format(m2.coef_[i] - module.coefs_[i+1], train_X.columns[i], m2.coef_[i], module.coefs_[i+1]) for i in range(len(train_X.columns))])
    # print(*[module.coefs_[i+1] - m2.coef_[i] for i in m2.coef_.size()], sep='\n')

    df1 = []
    # df2 = []
    for pr in np.arange(10.,101.,1.):
        samples = list()
        # s2 = list()
        for i in range(10):
            batch = train_X.sample(frac=pr/100)
            loss = module.fit(batch.to_numpy(), train_y[batch.index].to_numpy()).loss(test_X.to_numpy(), test_y.to_numpy())
            # l2 = mean_squared_error(test_y.to_numpy(), m2.fit(batch.to_numpy(), train_y[batch.index].to_numpy()).predict(test_X.to_numpy()))
            samples.append(loss)
            # s2.append(l2)
        df1.append([pr, np.mean(samples), np.std(samples)])
        # df2.append([pr, np.mean(s2), np.std(s2)])
    df1 = pd.DataFrame.from_records(df1, columns=["p", "mean", "std"])
    # df2 = pd.DataFrame.from_records(df2, columns=["p", "mean", "std"])
    P = df1["p"].to_numpy()
    # P2 = df2["p"].to_numpy()
    mean_loss = df1["mean"].to_numpy()
    # mean_loss2 = df2["mean"].to_numpy()
    std_loss = df1["std"].to_numpy()
    # std_loss2 = df2["std"].to_numpy()

    go.Figure([go.Scatter(x=P, y=mean_loss, mode="markers+lines", name="Mean loss", line=dict(dash="dash"),
                          marker=dict(color="green", opacity=.7)),
               go.Scatter(x=P, y=mean_loss - 2 * std_loss, fill=None, mode="lines",
                          line=dict(color="lightgrey"), showlegend=False),
               go.Scatter(x=P, y=mean_loss + 2 * std_loss, fill='tonexty', mode="lines",
                          line=dict(color="lightgrey"), showlegend=False),
               # go.Scatter(x=P2, y=mean_loss2, mode="markers+lines", name="Mean loss", line=dict(dash="dash"),
               #            marker=dict(color="red", opacity=.7)),
               # go.Scatter(x=P2, y=mean_loss2 - 2 * std_loss2, fill=None, mode="lines",
               #            line=dict(color="lightblue"), showlegend=False),
               # go.Scatter(x=P2, y=mean_loss2 + 2 * std_loss2, fill='tonexty', mode="lines",
               #            line=dict(color="lightblue"), showlegend=False)
               ],
              layout=go.Layout(title=r"$\text{(4) Average Loss } MSE \text{ As Function of } p%$",
                               xaxis_title=r"$p%$ - Training Sample Percentage",
                               yaxis_title=r"Average Loss")).show()

