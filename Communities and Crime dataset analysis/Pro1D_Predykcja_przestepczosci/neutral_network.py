from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
import random

def neural_network(X_train, X_test, y_train, y_test, X, y):

    regressionModel = MLPRegressor(hidden_layer_sizes=(90, 40, 10), activation='logistic', alpha=0.001,
                           learning_rate_init=0.001,
                           learning_rate='adaptive', solver='adam', max_iter=10000, early_stopping=True, verbose=False)
    regressionModel.fit(X_train, y_train)
    print('Neural network prediction: ', regressionModel.score(X_test, y_test))

    scores = cross_val_score(regressionModel, X.values, y.values.ravel(), cv=10, scoring='neg_mean_squared_error')
    scores2 = cross_val_score(regressionModel, X.values, y.values.ravel(), cv=10, scoring='r2')

    print('Neural network cross-validation mean score mse is: ', scores.mean(), scores.std())
    print('Neural network cross-validation mean score r2 is: ', scores2.mean(), scores2.std())

def n_n_parameter_search(X_train, y_train):

        regressionModel = MLPRegressor(learning_rate='adaptive', max_iter=5000, random_state=42, early_stopping=True)
        params = {
            'activation': ['logistic', 'relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'hidden_layer_sizes': [(random.randrange(20, 100), random.randrange(10, 50), random.randrange(5, 20)) for i
                                   in
                                   range(40)],
            'solver': ['sgd', 'adam']
        }

        regressionModelRandom = RandomizedSearchCV(estimator=regressionModel, param_distributions=params, n_iter=120, cv=4, verbose=5,
                                            n_jobs=-1, random_state=42)
        regressionModelRandom.fit(X_train, y_train)

        print(regressionModelRandom.best_params_)
        print(regressionModelRandom.best_estimator_)
