########## LIGHTGBM
# Verwendet histogram-basierenden Algorithmus,
# welcher kontinuierliche Features in Buckets plaziert
# => vs (standard)xbgoost: optimiert Geschwindigkeit und Speichernutzung
# Läßt den Baum nicht level-wise sondern tree-wise wachsen
# => erzeugt geringeren loss
# Hat besseren Algorithmus für Splits bei kategorischen Features
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from numpy import loadtxt
import matplotlib.pyplot as plt

#### Daten laden
filename = '~/Documents/Data/housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv(filename, delim_whitespace=True, names=names)

#### Test/Train Datensätze erzeugen
array = dataset.values
X = array[:,0:13]
Y = array[:,13]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)


#### Train Model
evals_result = {}
gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=20)

gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=5
        )

#### Vorhersage
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

#### Evaluation & Feature Importance
print('RMSE:', mean_squared_error(y_test, y_pred) ** 0.5)
print('R2:', r2_score(y_test, y_pred))
print('Feature importances:', list(gbm.feature_importances_))



#### Grid Search
estimator = lgb.LGBMRegressor(num_leaves=31)
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40, 50, 60, 70, 80, 90] }

gbm = GridSearchCV(estimator, param_grid)
gbm.fit(X_train, y_train)

print('Best parameters: ', gbm.best_params_)
print('Best Score:', gbm.best_score_)
y_pred = gbm.predict(X_test)
print('RMSE:', mean_squared_error(y_test, y_pred) ** 0.5)
print('R2:', r2_score(y_test, y_pred))



#### Visualisieren
## Metrik während des Trainings
ax = lgb.plot_metric(gbm.evals_result_, metric='l1')

## Feature Importance
ax = lgb.plot_importance(gbm, max_num_features=10)

## Bäume Plotten
ax = lgb.plot_tree(gbm, tree_index=1, figsize=(20, 8), show_info=['split_gain'])
graph = lgb.create_tree_digraph(gbm, tree_index=1, name='Tree84')
graph.render(view=True)
