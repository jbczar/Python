from datetime import time

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt





df = pd.read_csv('Credit_card_data.csv', delimiter=';')
num_records, num_features = df.shape
print(f'Liczba rekordów: {num_records}, Liczba cech: {num_features}')
print("")

# Rozkład kategorii
category_distribution = df['default payment next month'].value_counts()
print('Rozkład kategorii:')
print(category_distribution)
print("")

X = df.drop(columns=['ID', 'default payment next month'])
y = df['default payment next month']
print("")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    # Trenowanie klasyfikatora i predykcja na zbiorze testowym
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Obliczenie miar
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    return accuracy, f1, roc_auc


# Optymalizuje hyperparametry dla DT
dt_params = {'max_depth': [2, 4], 'criterion': ['gini', 'entropy']}
dt_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42),
                              dt_params, scoring='roc_auc', cv=5)
dt_grid_search.fit(X_train, y_train)

print("Best parameters for Decision Tree:", dt_grid_search.best_params_)
print("Best ROC AUC Score for Decision Tree:", dt_grid_search.best_score_)
print("")

# Zadanie 5: Użyj GridSearchCV dla optymalizacji hyperparametrów DT
dt_params_task5 = {'max_depth': [2, 4, 6, 8, 10, 12], 'criterion': ['gini', 'entropy']}
dt_grid_search_task5 = GridSearchCV(DecisionTreeClassifier(random_state=42),
                                    dt_params_task5, scoring='roc_auc', cv=5)
dt_grid_search_task5.fit(X_train, y_train)

print("Best parameters for Decision Tree (After GridSearchCV):", dt_grid_search_task5.best_params_)
print("Best ROC AUC Score for Decision Tree (After GridSearchCV):", dt_grid_search_task5.best_score_)
print("")

# Tworzenie klasyfikatora Random Forest i wizualizacja
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)


# Wizualizacja ważności cech w Random Forest
feature_importances = rf_classifier.feature_importances_
features = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.show()




# Under-sampling
X_train_under, y_train_under = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)

# Over-sampling
X_train_over, y_train_over = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)

# SMOTE
X_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Trenuje DT i RF na zbalansowanych zbiorach

dt_under_classifier = Pipeline([('under_sampler', RandomUnderSampler(random_state=42)),
                                ('classifier', DecisionTreeClassifier(random_state=42))])
rf_under_classifier = Pipeline([('under_sampler', RandomUnderSampler(random_state=42)),
                                ('classifier', RandomForestClassifier(random_state=42))])

dt_over_classifier = Pipeline([('over_sampler', RandomOverSampler(random_state=42)),
                               ('classifier', DecisionTreeClassifier(random_state=42))])
rf_over_classifier = Pipeline([('over_sampler', RandomOverSampler(random_state=42)),
                               ('classifier', RandomForestClassifier(random_state=42))])

dt_smote_classifier = Pipeline([('smote', SMOTE(random_state=42)),
                                ('classifier', DecisionTreeClassifier(random_state=42))])
rf_smote_classifier = Pipeline([('smote', SMOTE(random_state=42)),
                                ('classifier', RandomForestClassifier(random_state=42))])

# Testowanie modeli na zbiorze testowym
# Decision Tree
print("Decision Tree without balancing:")
accuracy, f1, roc_auc = evaluate_classifier(DecisionTreeClassifier(random_state=42), X_train, y_train, X_test, y_test)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("")

print("Decision Tree with Under-sampling:")
accuracy, f1, roc_auc = evaluate_classifier(dt_under_classifier, X_train_under, y_train_under, X_test, y_test)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("")

print("Decision Tree with Over-sampling:")
accuracy, f1, roc_auc = evaluate_classifier(dt_over_classifier, X_train_over, y_train_over, X_test, y_test)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("")

print("Decision Tree with SMOTE:")
accuracy, f1, roc_auc = evaluate_classifier(dt_smote_classifier, X_train_smote, y_train_smote, X_test, y_test)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("")

# Random Forest
print("Random Forest without balancing:")
accuracy, f1, roc_auc = evaluate_classifier(RandomForestClassifier(random_state=42), X_train, y_train, X_test, y_test)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("")

print("Random Forest with Under-sampling:")
accuracy, f1, roc_auc = evaluate_classifier(rf_under_classifier, X_train_under, y_train_under, X_test, y_test)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("")

print("Random Forest with Over-sampling:")
accuracy, f1, roc_auc = evaluate_classifier(rf_over_classifier, X_train_over, y_train_over, X_test, y_test)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("")

print("Random Forest with SMOTE:")
accuracy, f1, roc_auc = evaluate_classifier(rf_smote_classifier, X_train_smote, y_train_smote, X_test, y_test)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("")



