########## ML - REGULARISIERUNG
# Regulization = Hinzfügen von Penalty-Term zur Loss-Function zur Bestrafung von großen Koeffizienten
#  => Drei Möglichkeiten: Ridge(L2), Lasso(L1), Elastic net(L1 und L2)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#### Daten vorbereiten
boston = pd.read_csv('~/Documents/Data/Boston.csv')
boston.columns = [c.upper() for c in boston.columns]
X = boston.drop('MEDV',axis=1)
y = boston['MEDV']
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)





#### 1. Ridge Regression:
# Loss Function = OLS loss function + alpha * Summe der quadrierten Koeffizienten
# Regulization Term ist die L2-Norm der Koeffizienten => L2 Regularisierung
# => alpha = 0 => normales OLS (kann zu overfitting führen)
# => Modelle mit hohen (negativen) Koeffizienten werden bestraft
# zu hohes alpha => underfitting
ridge = Ridge(alpha=0.1,normalize=True)
_ = ridge.fit(X_train,y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test,y_test) # hier allerdings mit 0.69 etwas schlechter als normal






#### 2. Lasso Regression:
# Loss Function = OLS loss function + alpha * summe der absoluten Werte der Koeffizienten
# Regulization Term = L1-Norm der Koeffizienten => L1 Regularisierung
# => impliziete Featureauswahl (schrumpft Koeffizienten auf 0 (vs nahe 0 bei Ridge))
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
lasso = Lasso(alpha=0.1,normalize=True)
fit = lasso.fit(X_train,y_train)
Lasso_pred = lasso.predict(X_test)
lasso.score(X_test,y_test)

## => nimmt feature selection vor
names = boston.drop('MEDV',axis=1).columns
lasso_coef = fit.coef_
_ = plt.scatter(range(len(names)),lasso_coef)
_ = plt.xticks(range(len(names)),names,rotation=60)
_ = plt.ylabel('Coefficients')
plt.show()





#### 3. Elastic net Regulization MIT CV
# Penalty term: Lineare Kombination aus L1 und L2: a * L1 + b * L2
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)

l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}
elastic_net = ElasticNet()
gm_cv = GridSearchCV(elastic_net,param_grid,cv=5)
_ = gm_cv.fit(X_train,y_train)
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))
