from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
import random


def non_linear_regression(X_train, X_test, y_train, y_test, X, y):

    regressionModel = SVR()
    regressionModel.fit(X_train, y_train)
    print('Non-linear regression prediction: ', regressionModel.score(X_test, y_test))

    scores = cross_val_score(regressionModel, X.values, y.values.ravel(), cv=10, scoring='neg_mean_squared_error')
    scores2 = cross_val_score(regressionModel, X.values, y.values.ravel(), cv=10, scoring='r2')

    print('Non-Linear regression cross-validation mean score mse is: ', scores.mean(), scores.std())
    print('Non-Linear regression cross-validation mean score r2 is: ', scores2.mean(), scores2.std())
