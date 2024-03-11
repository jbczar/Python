from pyparsing import results
from sklearn.model_selection import train_test_split

import neutral_network as nn
import linear as ln
import nonLinear as nln
import decision_tree as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main():
    # zbiory danych
    orginal_data, cleaned_data , mean_data, median_data = prep_data()

    # selekcja
    X_temp, y_temp = prep_dataset(cleaned_data)
    column_names = feature_selector_correllation(X_temp, y_temp,threshold=0.40)

    print(len(column_names))
    print("")
    for i in range(len(column_names)):
        print(column_names[i])

    column_names = np.append(column_names, 'ViolentCrimesPerPop')
    cleaned_data_filtered = cleaned_data[column_names]


    #heat mapa pokazująca korelacje
    selected_correlation = cleaned_data[column_names].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(selected_correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Korelacje po selekcji cech - ' + str(cleaned_data))
    plt.show()


    correlations_with_target = selected_correlation['ViolentCrimesPerPop']
    positive_correlations = correlations_with_target[(correlations_with_target >= 0)]
    top_positive_correlations = positive_correlations.sort_values(ascending=False).head(5)
    top_positive_correlations_df = pd.DataFrame(
        {'Feature': top_positive_correlations.index, 'Correlation': top_positive_correlations.values})

    top_positive_features = cleaned_data[top_positive_correlations.index]


    plt.figure(figsize=(12, 6))
    for feature in top_positive_features.columns:
        plt.bar(feature, top_positive_features[feature].corr(cleaned_data['ViolentCrimesPerPop']))


    plt.title('Positive Correlations with ViolentCrimesPerPop')
    plt.xlabel('Features')
    plt.ylabel('Correlation')
    plt.xticks(rotation=45, ha='right')
    plt.show()


    top_features = top_positive_features.columns[:5]
    plt.figure(figsize=(18, 12))
    for i, feature in enumerate(top_features, 1):
        plt.subplot(2, 3, i)  # Adjust the subplot layout according to the number of features
        sns.scatterplot(x=feature, y="ViolentCrimesPerPop", data=cleaned_data)
        plt.title(f'Scatter Plot for {feature}')

    plt.tight_layout()
    plt.show()





    data_sets_names = ["cleaned_data", "cleaned_data_filtered", "mean_data", "median_data"]
    data_sets = [cleaned_data, cleaned_data_filtered, mean_data, median_data]


    for index in range(len(data_sets)):
        print('\n------------------------------\nData set name:', data_sets_names[index])
        # inicjacja X i y
        inputs = data_sets[index]
        X, y = prep_dataset(inputs)

        #zbiory testowe i treningowe podział
        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.25)
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        nn_result = nn.neural_network(X_train, X_test, y_train, y_test, X, y)   # siec neuronowa
        dt_result = dt.decision_tree_regressor(X_train, X_test, y_train, y_test, X, y) # Drzewo regresyjne
        ln_result = ln.linear_regression(X_train, X_test, y_train, y_test, X, y) # regresja liniowa
        nln_result = nln.non_linear_regression(X_train, X_test, y_train, y_test, X, y)  # regresja nieliniowa



def find_parameters(X_train, y_train):
    nn.n_n_parameter_search(X_train, y_train) #random search


def prep_data():

    # Dane surowe
    orginal_data = pd.read_csv('crimedata.csv', sep=';')
    orginal_data = orginal_data.iloc[:, 5:]

    # Dane bez kolum z pustymi wartościami
    cleaned_data = pd.read_csv('crimedata.csv', sep=';')
    cleaned_data = cleaned_data.replace('?', np.NaN)
    cleaned_data = cleaned_data.dropna(axis=1)
    cleaned_data = cleaned_data.iloc[:, 5:]

    # puste warotściami wypełonymi średnią
    mean_data = pd.read_csv('crimedata.csv', sep=';')
    mean_data = mean_data.iloc[:, 5:]
    mean_data = mean_data.replace('?', np.NaN)
    mean_data = mean_data.apply(pd.to_numeric)
    mean_data = mean_data.fillna(mean_data.mean())

    # puste warotściami wypełonymi medianą
    median_data = pd.read_csv('crimedata.csv', sep=';')
    median_data = median_data.iloc[:, 5:]
    median_data = median_data.replace('?', np.NaN)
    median_data = median_data.apply(pd.to_numeric)
    median_data = median_data.fillna(median_data.median())

    return orginal_data, cleaned_data, mean_data, median_data


def prep_dataset(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    return X, y


def feature_selector_correllation(X, y, threshold=0.40):

    y = y.values.ravel() if isinstance(y, pd.DataFrame) else y.ravel()
    correlations = X.corrwith(pd.Series(y))
    selected_features = correlations[abs(correlations) > threshold].index.tolist()

    return selected_features

if __name__ == '__main__':
    main()