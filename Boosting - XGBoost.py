########## eXtreme Gradient Boosting
# Geeignete Tuning-Parameter Kombinationen:
# - Anzahl der Bäume: n_estimators, max_depth
# - Learning Rate und Anzahl der Bäume:  learning_rate, n_estimators
# - Row & column subsampling rates: subsample, colsample_bytree, colsample_bylevel
#
# Startwerte:
# learning_rate: 0.1 oder niedriger
#      => kleine learning-rate benötigt dann mehr trees
# tree_depth: 2-8
#      => idr kein Vorteil durch noch mehr
# subsample: 30%-80% des Trainingsets
#      => vergleichen mit 100% für kein sampling
#
# Strategie:
# Anfangen mit default Konfiguration und dann learning-curve von train & test angucken
# => bei Overfitting:  learning-rate runter und/oder Anzahl der Bäume hoch
# => bei Underfitting: learning-rate hoch   und/oder Anzahl der Bäume runter
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

#### Daten laden
%cd ~
dataset = loadtxt('Documents/Data/pima-indians-diabetes.data.csv', delimiter=",")

#### Test/Train Datensätze erzeugen
X = dataset[:,0:8]
Y = dataset[:,8]
seed = 0
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


#### Modell trainieren
## Einfach...
#params = { "n_estimators": 400, 'tree_method':'gpu_hist', 'predictor':'gpu_predictor' }
#model = XGBClassifier(**params)

model = XGBClassifier()
model.fit(X_train, y_train)

## ...oder mit early stopping
# XGBoost kann während des Trainings die performance auf test-set messen und reporten
# und early stopping beim Training, wenn eine Anzahl von Iterationen
# keine Verbesserung auf dem test-set zu sehen ist
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)


## Hyperparamter Tuning
clf2 = XGBClassifier()
## Parameter-Grid erstellen
n_estimators = [50, 100, 150, 200]
max_depth = [2, 4, 6, 8]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
## Grid Search mit 10-fold Cross Validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(clf2, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
model2 = grid_search.fit(X, Y)
# Beste Kombination
print("Best: %f using %s" % (model2.best_score_, model2.best_params_))

means = model2.cv_results_['mean_test_score']
stds = model2.cv_results_['std_test_score']
params = model2.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


#### Vorherhsagen machen
predictions = model.predict(X_test)
predictions2 = model2.predict(X_test)

#### Evaluieren des Modells
accuracy = accuracy_score(y_test, predictions)
accuracy2 = accuracy_score(y_test, predictions2)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Accuracy: %.2f%%" % (accuracy2 * 100.0))


#### Feature Importance anzeigen
print(model.feature_importances_)
plot_importance(model)
pyplot.show()
