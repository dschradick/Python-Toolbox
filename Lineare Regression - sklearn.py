########## LINEARE REGRESSION - SKLEARN
import numpy as np
import pandas as pd
from ggplot import mtcars
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import linear_model
matplotlib.style.use('ggplot')


#### Daten betrachten
mtcars.plot(kind="scatter", x="wt", y="mpg", figsize=(9,9), color="black");


#### Lineare Regression durchführen
regression_model = linear_model.LinearRegression()
X = mtcars.drop(["mpg","name"], axis=1)
y = mtcars["mpg"]
## Modell fitten
regression_model.fit(X, y)


#### Ergebnis
## Intercept und Koeffizienten
regression_model.intercept_
regression_model.coef_

## R^2 des Modells
regression_model.score(X, y)


#### Mini- Diagnostik
# Residuen berechnen
residuals = mtcars["mpg"] - train_prediction
residuals.describe()
plt.scatter(residuals.index,residuals)

## QQ-Plot
plt.figure(figsize=(9,9))
stats.probplot(residuals, dist="norm", plot=plt)


#### Vorhersagen machen
train_prediction = regression_model.predict(X)
