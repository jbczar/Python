import random

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def linear_regression(X_train, X_test, y_train, y_test, X, y):

    regressionModel = LinearRegression()

    regressionModel.fit(X_train, y_train)
    print('Linear regression prediction: ', regressionModel.score(X_test, y_test))

    scores = cross_val_score(regressionModel, X.values, y.values.ravel(), cv=10, scoring='neg_mean_squared_error')
    scores2 = cross_val_score(regressionModel, X.values, y.values.ravel(), cv=10, scoring='r2')
    print('Linear regression cross-validation mean score mse is: ', scores.mean(), scores.std())
    print('Linear regression cross-validation mean score r2 is: ', scores2.mean(), scores2.std())
