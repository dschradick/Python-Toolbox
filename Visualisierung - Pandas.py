########## VISUALISIERUNG - PANDAS
# Generell: nur für quickplots
# => aufwendigere Visualisierungen mit seaborn

# https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.plot.html
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc("axes.spines", top=False, right=False)  # despine() als default

mtcars = pd.read_csv('~/Documents/Data/mtcars.csv', index_col='name')
diamonds = pd.read_csv('~/Documents/Data/diamonds.csv')


diamonds.head(1)

## Bessere Farben
plt.style.use('seaborn-whitegrid')  # alternativ 'seaborn-whitegrid', 'ggplot', 'bmh','dark_background'

## Einfacher Plot
# 1.
plt.figure(figsize=(8,8))
mtcars.mpg.plot()                  # =...
plt.plot(mtcars.index,mtcars.mpg)
# 2.
ax = diamonds.carat.plot(kind='hist', title='Titel')
ax.set_xlabel("x label ($)")
ax.set_xlabel("y label ($)")

## Daten vor plotten auswählen
diamonds.query('carat > 3.5 & depth < 70')
#df.query('2014 < date < 20150131')

### GRUPPEN
mtcars.groupby('cyl')['hp'].mean().plot(kind='barh')                      # Mittelwerte der Gruppen
mtcars.groupby('cyl')['hp'].count().sort_values().plot(kind='barh')       # Anzahl in den Gruppen
# Untergruppen -  dodged oder verschiedene Diagramme
mtcars.groupby(['cyl','am'])['hp'].mean().unstack().plot(kind='barh')
mtcars.groupby(['cyl','am'])['hp'].mean().unstack().plot(kind='barh',subplots=True)

## Subplots
diamonds[['price','depth']].plot(kind='box', subplots=True, layout=(1,2))


## Grundform
diamonds.price.plot(kind='hist', bins=20)
diamonds.plot(x='carat',y='price',kind='scatter', s=10)
### Argument 'kind' für plot
# => für default-Plotting
# bar / barh:         bar plots
# hist:               histogram
# box:                boxplot
# kde / density':     density plots
# area:               area plots
# scatter:            scatter plots
# hexbin:             hexagonal bin plots
# pie:                pie plots
#  => häufig kombiniert mit crosstable


## Histogram
# PDF: kind='hist', normed=True
# CDF: kind='hist', normed=True, cumulative=True
diamonds.price.hist()                 # äquivalent zu...
diamonds.price.plot(kind='hist')
diamonds.price.plot(kind='hist',figsize=(8,8), color="blue", bins=50, range= (0,10000))

## Density
diamonds.price.plot.kde()             # äquivalent zu...
diamonds.price.plot(kind="density")

## Barplot
pd.crosstab(mtcars.cyl,mtcars.carb,margins=True).plot(kind='barh')
carat_table = pd.crosstab(index=diamonds["clarity"], columns="count")
carat_table.plot(kind='bar')
# Barplot mit zweiter Variablen
carat_table = pd.crosstab(index=diamonds["clarity"], columns=diamonds["color"])
carat_table.plot(kind="bar", figsize=(8,8), stacked=True)
carat_table.plot(kind="bar", figsize=(8,8), stacked=False)

## Boxplots
diamonds.carat.plot(kind='box')
diamonds.boxplot(column="price", by= "clarity", figsize= (8,8))

## Scatterplot
# => Kein mapping von Punkten auf Farbe => seaborn benutzen
diamonds.plot(kind="scatter", x="carat", y="price")
diamonds.plot(kind="scatter", x="carat", y="price", s=10, figsize=(10,10), ylim=(0,20000))
diamonds.plot.scatter(x="carat", y="price", c=diamonds['carat'], s=diamonds['carat']*2);  # color + size

## Scattermatrix
scatter_matrix(diamonds.iloc[1:100,:], alpha=0.2, figsize=(6, 6), diagonal='kde');


## Line
years = [y for y in range(1950,2016)]
readings = [(y+np.random.uniform(0,20)-1900) for y in years]
time_df = pd.DataFrame({"year":years, "readings":readings})
time_df.plot(x="year", y="readings", figsize=(9,9))


## Einfacher Subplot
# subplot(nrow,ncols,index):
plt.subplot(2,2,1); mtcars.mpg.plot()
plt.subplot(2,2,2); mtcars.hp.plot(kind='hist')
plt.subplot(2,2,3); sns.distplot(mtcars.hp,kde=False)
plt.subplot(2,2,4); sns.countplot(x='cyl',data=mtcars)
plt.tight_layout()

# Unterer nimmt beide plätze
plt.subplot(2,2,1); mtcars.mpg.plot()
plt.subplot(2,2,2); mtcars.hp.plot(kind='hist')
plt.subplot(2,1,2); sns.distplot(mtcars.hp,kde=False)
