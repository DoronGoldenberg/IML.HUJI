import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename,parse_dates=["Date"])
    df = df.drop(index=df[df.Temp < -30].index).reset_index(drop=True)
    # lowest recorded temperature in Israel, Jordan, South Africa and The Netherlands are around
    # -14, -16, -20, -27 respectivly.
    df['DayOfYear'] = df['Date'].dt.day_of_year
    df['TempNormal'] = (df['Temp']-df['Temp'].mean())/ df['Temp'].std()
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    IL = df[df.Country == 'Israel'].copy()
    IL['Year'] = IL['Year'].astype(str)
    px.scatter(IL, x='DayOfYear', y='Temp', color='Year', title='Daily Temperature in Israel').show()
    # 2 pics, so polynom of degree 3.
    IL_month = IL.groupby(['Month'])['Temp'].std().reset_index()
    px.bar(IL_month, x='Month', y='Temp', title='Standard Deviation of Temperature in Israel per Month').show()
    # The algoritme will not work equaly for each month, for example in 7,8 the daily temp deviate less, there for
    # it will be closuer to the algo prediction.

    # Question 3 - Exploring differences between countries
    df_group = df.groupby(['Country','Month'])['Temp'].agg([np.mean,np.std]).reset_index()
    px.line(df_group, x='Month', y='mean', error_y='std', color='Country',
            title='Average Daily Temperature per Month').show()
    # All the countryies share simular patern, but countries in the lower himsfir is invers. moudle fitted for il
    # will probbly work ok for countries in the same latatuid.

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(IL['DayOfYear'], IL['TempNormal'], train_proportion=0.75)
    loss_df = []
    for k in np.arange(1, 11):
        module = PolynomialFitting(k)
        module.fit(train_X.to_numpy(), train_y.to_numpy())
        loss = module.loss(test_X.to_numpy(), test_y.to_numpy())
        loss_df.append([k, loss])
    loss_df = pd.DataFrame.from_records(loss_df, columns=["k", "loss"])
    px.bar(loss_df, x='k', y='loss', title='Test Error Value as Function of the polynom degree k').show()
    [print('degree {}: {:.2f}'.format(row['k'], row['loss'])) for _, row in loss_df.iterrows()]

    # Question 5 - Evaluating fitted model on different countries
    module = PolynomialFitting(3)
    module.fit(IL['DayOfYear'], IL['TempNormal'])

    tmp = IL[['DayOfYear', 'TempNormal']].copy()
    tmp['pred'] = module.predict(IL['DayOfYear'])
    tmp = tmp.sort_values('DayOfYear')
    px.line(tmp, x='DayOfYear', y=['TempNormal', 'pred'], title='Module Daily Temperature Prediction').show()

    loss_df = []
    for country, df_c in df.groupby(['Country']):
        loss = module.loss(df_c['DayOfYear'].to_numpy(), df_c['TempNormal'].to_numpy())
        loss_df.append([country, loss])
    loss_df = pd.DataFrame.from_records(loss_df, columns=["Country", "loss"])
    px.bar(loss_df, x='Country', y='loss', title='Test Error Value as Function of the Country').show()
