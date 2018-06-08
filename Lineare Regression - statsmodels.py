########## LINEARE REGRESSION - STATSMODELS
import numpy as np
from statsmodels.formula.api import ols
from ggplot import diamonds
import seaborn as sns
import matplotlib.pyplot as plt

#### Datensatz verkleinern
np.random.seed(0)
indices = np.random.choice(diamonds.shape[0],size=50)
diamonds = diamonds.iloc[indices]
diamonds.head()

#### Beziehungen visuell betrachten
sns.pairplot(diamonds.iloc[1:500], vars=['price', 'carat','depth'], hue='cut', kind='reg')
sns.jointplot(x='carat', y='price', data=diamonds)
## Auf Interaktionen überprüfen
sns.lmplot(x='carat', y='price', hue='cut', data=diamonds)



#### Einfache Lineare Regression
# In statsmodels:
# - y = endogenous
# - x = exogenous
sns.lmplot(x='carat', y='price', data=diamonds)
model = ols("price ~ carat", diamonds).fit()
model.summary()



#### Muliple Lineare Regression
## Mit kategorischer Variable - C() forciert kategorische Variable
## durch np.power usw. Variablen transformieren
## Interaktionen:  *, :
# oder: disp + I(disp**2)
f = "price ~ np.log(carat) + np.power(depth,2) + C(cut) + depth:carat"
model = ols("price ~ carat + depth + cut", diamonds).fit()

model.summary()
model.rsquared    # R^2
model.params      # Koeffizienten
model.bse         # Standard-Errors
model.conf_int()  # Konfidenz-Intervalle



### broom::augment
summary = model.get_influence().summary_frame()
summary[['dfb_carat','dfb_depth','cooks_d','standard_resid']]



### Diagnostik
# http://www.statsmodels.org/dev/examples/notebooks/generated/regression_plots.html
import statsmodels.api as sm
##  Residuenplot
sns.residplot(x="carat", y="price", data=diamonds);

## Fit Plot
fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.plot_fit(model, "carat", ax=ax)

## Influence Plot
fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.influence_plot(model, ax=ax, criterion="cooks")

## Partial Regression Plot
# Effekt von hinzugefügter Variable
fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.plot_partregress("price", "carat", ["carat", "depth"], data=diamonds, ax=ax)

fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.plot_partregress("price", "carat", ["carat"], data=diamonds, ax=ax)


### Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif



## ANOVA: Unterschied Premium und Very Good?
#mittels vector of contrast
len(model.params)
model.f_test([0, 0, 0, 1, -1, 0, 0])
