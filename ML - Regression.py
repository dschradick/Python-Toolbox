########## ML - REGRESSION
# 1. Datensatz textuell / visuell beschreiben
# 2. Datensatz vorbereiten
# 3. Spotchecking
# 4. Optimierung
# 5. Vorhersagen machen
# 6. Modell auf gesamten Datensatz trainieren
# 7. Finales Modell speichern (und laden)
# ==> 3. & 4. jeweils mit einfachen/ensemble Algorithmen
#     und normales / standarisiertem Datensatz
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.externals.joblib import dump, load
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#### Daten aus CSV laden
filename = '~/Documents/Data/housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = read_csv(filename, delim_whitespace=True, names=names)


### Mini-EDA
dataset.shape
dataset.dtypes
dataset.head(5)
set_option('precision', 1)
dataset.describe()
## Korrelationsmatrix
set_option('precision', 2)
dataset.corr(method='pearson')


### Daten (kurz) visuell beschreiben
_ = dataset.hist() ; plt.show()
# Dichte-Funktion
_ = dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False)
plt.show()
# Box-Plots
_ = dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
plt.show()
# Ziel-Variable
_ = dataset.MEDV.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
plt.show()
# Scatterplot Matrix
_ = scatter_matrix(dataset)
plt.show()
# Korrelationsmatrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = np.arange(0,14,1)
_ = ax.set_xticks(ticks) ; _ = ax.set_yticks(ticks)
_ = ax.set_xticklabels(names) ; _ = ax.set_yticklabels(names)
plt.show()



### Daten vorbereiten
## Train- / Testset erzeugen
array = dataset.values
X = array[:,0:13]
Y = array[:,13]
validation_size = 0.20
seed = 0
	X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

### Spot-Checking der Algorithmen
num_folds = 10
seed = 0
scoring = 'neg_mean_squared_error'

# Zu evaluierende Algorithmen hinzufügen
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

## Ausgewählten Modelle mit k-fold evaluieren & textuell ausgeben
# => Baseline erzeugen
results = []; names = []
for name, model in models:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

## Ergebnisse visuell anzeigen
fig = plt.figure()
fig.suptitle('Algorithmen Vergleich')
ax = fig.add_subplot(111)
_ = plt.boxplot(results)
_  = ax.set_xticklabels(names)
plt.show()


### Evaluieren mit standarisiertem Datenset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
results = []
names = []
## Evaluieren & textuell anzeigen
for name, model in pipelines:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(abs(cv_results))
	names.append(name)
	msg = "%s: %f (%f)" % (name, abs(cv_results.mean()), cv_results.std())
	print(msg)

# Ergebnisse grafisch anzeigen
fig = plt.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
_ = plt.boxplot(results)
_ = ax.set_xticklabels(names,rotation=90)
plt.show()
## ====> KNN auswählen als Algorithmus

### Optimizerung mittels Grid-GridSearch
# ==> der Algorithmus, welcher sich beim Spot-Checking als bester
#     herausgestellt herausgestellt, wird nun optimiert
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = np.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
## Alternativ: RandomizedSearchCV
#grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring=scoring, cv=kfold, n_iter=5, random_state=7)
grid.fit(X, Y)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (abs(mean), stdev, param))


### Standard Ensembles
# z.B. Random Forest, etc
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor())])))
results = []
names = []
## Evaluieren und Ergbnisse textuell
for name, model in ensembles:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(abs(cv_results))
	names.append(name)
	msg = "%s: %f (%f)" % (name, abs(cv_results.mean()), cv_results.std())
	print(msg)

## Ergebnisse visuell
fig = plt.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
_ = plt.boxplot(results)
_ = ax.set_xticklabels(names,rotation=90)
plt.show()


### Optimierung
# von GBM in diesem Fall
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=np.array([50,100,150,200,250,300,350,400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
grid_result.best_score_
grid_result.best_estimator_.n_estimators

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


#### Vorhersagen
# Modell vorbereiten
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
model.fit(rescaledX, Y_train)
# Transformation des Validierungsset
rescaledValidationX = scaler.transform(X_validation)
# Vorhersage & MSE berechnen
predictions = model.predict(rescaledValidationX)
mean_squared_error(Y_validation, predictions)
# ====> MSE: 16.3

#### Modell speichern
filename = 'final_model.sav'
dump(model, filename)
# später wieder laden mitk
loaded_model = load(filename)
predictions = model.predict(rescaledValidationX)
mean_squared_error(Y_validation, predictions)
