########## ML - HYPERPARAMETER TUNING
# Hyperparameter = Parameter, die vor fitting des Algorithmus gesetzt werden müssen
#   => werden nicht während des normalen fittings gelernt werden
# Bsp: KNN - k, Ridge/Lasso - alpha
# Tuning: verschiedene Werte für Hpyerparameter ausprobieren,
#         jeweils mit diesen fitten, performance messen, best-performing auswählen
#   => dabei immer CV benutzen! (ansonsten overfitting der hyperparameter)
# Gridsearch CV: verwendet Grid von Werten (geht auch in Pipeline)

#### Daten einlesen
import pandas as pd
import numpy as np
data = pd.read_csv('~/Documents/Data/german_credit_test.csv')
target = 'Creditability'
X = data.drop(target,axis=1)
y = data[target]



#### Normaler Grid-Search
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

param_grid = {'n_neighbors': np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn,param_grid, cv=5)
_  = knn_cv.fit(X,y)
knn_cv.best_params_
knn_cv.best_score_


# mit Logistischer Regression
from sklearn.linear_model import LogisticRegression
c_space = np.logspace(-5, 8, 15)
# C: kontrolliert inverse Regularisierungs-stärke
# => C zu groß => overfitting
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,param_grid,cv=5)
_  = logreg_cv.fit(X,y)
logreg_cv.best_estimator_
logreg_cv.best_score_





#### Randomized Search
# Geeignet, wenn viele optimale Hpyerparameter gleichzeitig gefunden werden sollen
# => nicht alle Werte werden ausprobiert, sondern Werte aus vorgegebenen Verteilungen genommen
# Beispiel: DecisionTreeClassifier
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

# Verteilungen für die Parameter festelegen
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree,param_dist,cv=5)
tree_cv.fit(X,y)
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
