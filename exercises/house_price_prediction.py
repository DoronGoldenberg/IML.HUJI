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

    # Drop unclassified data.
    df = df.drop(index=df[df.id == 0].index)
    df = df.dropna(subset=['price'])
    df = df.drop(index=df[df.price <= 0].index).reset_index()

    # In case yr_renovated or sqft_lot15 data is impossibly replaced it with yr_built or sqft_lot respectively.
    df.loc[df['yr_renovated'] == 0, 'yr_renovated'] = df.loc[df['yr_renovated'] == 0, 'yr_built']
    df.loc[df['sqft_lot15'] <= 0, 'sqft_lot15'] = df.loc[df['sqft_lot15'] <= 0, 'sqft_lot']

    # Convert lat, long cords to individual NxN bins. For each sample classify 1 in the bin the house lands in,
    # otherwise classify as 0.
    N = 5
    norm = (df[['lat', 'long']] - df[['lat', 'long']].min()) / (df[['lat', 'long']].max() - df[['lat', 'long']].min())
    A = np.zeros((norm.shape[0], N * N))
    bins = np.array([N, 1]) @ np.digitize(norm, np.linspace(0, 1, N-1)).T
    A[np.arange(norm.shape[0]), bins] = 1
    locations = pd.DataFrame(A, columns=['location {}'.format(i) for i in range(A.shape[1])])
    locations = locations.loc[:, (locations != 0).any(axis=0)]

    # One hot encode each zipcode address.
    zipcode = pd.get_dummies(df.zipcode, prefix='zipcode')

    # Separate prices from the dataset.
    prices = df['price']

    # Normilize each feature.
    prices = (prices-prices.mean())/ prices.std()
    discrete = df[["bedrooms", "bathrooms", "floors", "waterfront", "view", "condition", "grade"]]
    continuous = df[["sqft_living", "sqft_lot", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "sqft_living15", "sqft_lot15"]]
    discrete = (discrete-discrete.min())/(discrete.max()-discrete.min())
    continuous = (continuous-continuous.mean()) / continuous.std()

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
        if cov[0,1] == 0:
            p[feature] = 0
        else:
            p[feature] = cov[0,1] / np.sqrt(cov[0,0] * cov[1,1])
    correlation = sorted(p.items(), key=lambda item: np.abs(item[1]))
    print('most beneficial feature: {} (with personal correlation of {})'.format(correlation[-1][0],
                                                                                 correlation[-1][1]))
    print('least beneficial feature: {} (with personal correlation of {})'.format(correlation[0][0],
                                                                                  correlation[0][1]))
    px.scatter(X, x=correlation[0][0], y=y, title='Price as Function of ' + correlation[0][0],
               labels={'y': 'Price'}).show()
    px.scatter(X, x=correlation[-1][0], y=y, title='Price as Function of ' + correlation[-1][0],
               labels={'y': 'Price'}).show()


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

    loss_df = list()
    for percentage in np.arange(10.,101.,1.):
        samples = list()
        for i in range(10):
            batch = train_X.sample(frac=percentage/100)
            module.fit(batch.to_numpy(), train_y[batch.index].to_numpy())
            loss = module.loss(test_X.to_numpy(), test_y.to_numpy())
            samples.append(loss)
        loss_df.append([percentage, np.mean(samples), np.std(samples)])
    loss_df = pd.DataFrame.from_records(loss_df, columns=["Percentage", "Mean Loss", "STD Loss"])

    go.Figure([go.Scatter(x=loss_df['Percentage'], y=loss_df['Mean Loss'], mode="markers+lines", name="Mean Loss",
                          line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
               go.Scatter(x=loss_df['Percentage'], y=loss_df['Mean Loss'] - 2 * loss_df['STD Loss'], fill=None,
                          mode="lines", line=dict(color="lightgrey"), showlegend=False),
               go.Scatter(x=loss_df['Percentage'], y=loss_df['Mean Loss'] + 2 * loss_df['STD Loss'], fill='tonexty',
                          mode="lines", line=dict(color="lightgrey"), showlegend=False)],
              layout=go.Layout(title="Average MSE Loss as Function of Training Sample Percentage",
                               xaxis_title="Training Sample Percentage", yaxis_title="Average Loss")).show()
