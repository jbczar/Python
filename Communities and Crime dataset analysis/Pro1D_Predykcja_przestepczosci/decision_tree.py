from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
import random

def decision_tree_regressor(X_train, X_test, y_train, y_test, X, y):

    regressionModel = DecisionTreeRegressor()
    regressionModel.fit(X_train, y_train)
    print('Decision tree prediction: ', regressionModel.score(X_test, y_test))

    scores = cross_val_score(regressionModel, X.values, y.values.ravel(), cv=10, scoring='neg_mean_squared_error')
    scores2 = cross_val_score(regressionModel, X.values, y.values.ravel(), cv=10, scoring='r2')

    print('Decision tree cross-validation mean score mse is: ', scores.mean(), scores.std())
    print('Decision tree cross-validation mean score r2 is: ', scores2.mean(), scores2.std())