########## VISUALISIERUNG - SEABORN
# Wenn nur ein Wert für Kategorie vorhanden - dann der Wert
# => ansonsten wird eine Statistik berechnet (default ist mean)
import matplotlib.pyplot as plt
import seaborn as sns

from ggplot import mtcars, diamonds
from bokeh.sampledata.autompg import autompg as auto
from bokeh.sampledata.iris import flowers as iris
import scipy.stats as stats
import numpy as np
anscombe = sns.load_dataset("anscombe")


##### Besserer Stil
plt.style.use('ggplot')
mtcars.groupby('cyl').agg('count')


#### Größe, Achsen-Beschriftungen & Titel
plt.figure(figsize=(12,5))
ax = sns.countplot(y='cyl', order=mtcars.groupby('cyl')['name'].agg('count').sort_values().index, data=mtcars)
ax.set_xlabel('Number of Cylinders')
ax.set_ylabel('Number of Cars')
ax.set_title('Number of cylinders by number of cars')


mtcars.plot(kind='hist',subplots=True,layout=(4,4))

## Color Palette
#https://seaborn.pydata.org/tutorial/color_palettes.html
sns.set_palette("husl")

#### Achsen entfernen
sns.countplot(x="cut",data=diamonds)
sns.despine()

#### Sortieren des Plots
# den index neusortieren und als order mitgeben
# bei aggregierten index muss dieser per group-by erstellt werden
sorted_index = fp.groupby('dest')['dep_delay'].agg('median').sort_values().index
sns.boxplot(data=fp,y='dest',x='dep_delay',orient='h',order=sorted_index)



#### UNIVARIATE VERTEILUNGEN
#### Daten erzeugen
mu = 100; sigma = 15; x = mu + sigma * np.random.randn(1000)


### KONTINUIERLICHE DATEN
## Histogram & Density
sns.distplot(x)

## Legende: mehrere kurven
sns.distplot(x  , hist=False, label='Linie 1')
sns.distplot(x*2, hist=False, label='Linie 2')
plt.legend()

## Histogram
bins = [0,70,100,130,200]
sns.distplot(x, kde=False);                       # Einfach
sns.distplot(x, bins=20, kde=False);              # Bin Anzahl angeben
sns.distplot(x, bins=bins, kde=False);            # Explizite Bins
sns.distplot(x, rug=True);                        # Beobachtungen unten

## Density
sns.distplot(x, hist=False);                      # Einfach
sns.kdeplot(x);                                   # (wird von distplot verwendet)
sns.kdeplot(x, shade=True);                       # mattiert unter kurve
sns.kdeplot(x, bw=.2)                             # Bandwidth ~ binsize bei hist
## Fitting
sns.distplot(x, kde=False, fit=stats.norm);
sns.distplot(x, kde=False, fit=stats.gamma);


### KATEGORISCHE DATEN
diamonds['cut'] = diamonds['cut'].astype('category')
diamonds['cut'] = diamonds.cut.cat.reorder_categories(['Fair','Good','Very Good','Premium','Ideal'], ordered=True)
# => dann ordered seaborn auch automatisch

## Countplot
sns.countplot(y='origin', data=auto)
sns.countplot(y='origin', order=auto.origin.value_counts().index, data=auto),
sns.countplot(x='origin', hue='cyl',data=auto)

## Barplot
# Wenn nicht nur ein Wert => dann Konfidenzintervalle
sns.barplot(x='cyl',y='gear', data=mtcars)

## Boxplots
sns.boxplot(x='cyl',y='mpg', data=mtcars)
sns.boxenplot(x='cyl',y='mpg', data=mtcars)
## Verteilung innerhalb der Kategorie
sns.boxplot(x='cyl',y='mpg', hue='am', data=mtcars)


## Factorplots
# => heist jetzt catplot
sns.factorplot(x="yr", y="mpg", col="origin",  data=auto)
sns.factorplot(x="origin", y="mpg",data=auto,kind='bar')
sns.factorplot(x="origin", y="mpg",data=auto,kind='point')
sns.factorplot(x="origin", y="mpg",data=auto,kind='violin')


## Diese sind alle durch Factorplot mit kind='' darstellbar (hier nur größer)
sns.barplot(data=auto, x="origin", y="mpg")
sns.violinplot(x="origin", y="mpg",data=auto)
sns.violinplot(x='Survived',y='Age',scale='count',split=True,hue='Sex',inner='quartile',data=titanic) # Scale wichtig!!!

plt.show()

## Wide-form data
sns.catplot(kind='box', data=auto)
sns.violinplot(x=iris.species, y=iris.sepal_length)

### Facetgrid
## Einfach
g = sns.FacetGrid(mtcars, col="cyl",col_wrap=2)
g.map(sns.distplot, "mpg")
g.map(plt.hist, "mpg", alpha=.7)
## Weitere Möglichkeiten
g = sns.FacetGrid(mtcars, col="cyl", row='am', hue="origin")
g.map(plt.scatter, "hp", "mpg", alpha=.7)
g.add_legend();



#### Univariate Verteilungen

## Stripplot
import seaborn as sns
_ = sns.stripplot(y=x,jitter=False,size=3) # Jitter = horizontal auseinanderziehen, Size = Größe der Punkte
_ = sns.stripplot(y=x,orient='h')
# sns.stripplot(x='day' y='tip', data=tip) # bei dataframe
# ZUSAMMENHANG mit kategorischer Variable durch x=
plt.show()

## Swarmplot
# Macht Verteilung deutlicher: ordnet wiederholte Punkt automatisch horizontal an, um Overlap zu vermeiden
sns.swarmplot(y=x) # hue=kategorische Var möglich
plt.show()

## Violinplot
# Problem: Große Datenmenge => Boxplots, Violinplots
sns.violinplot(y=x)
plt.show()
sns.boxplot(y=x)
plt.show()

## ÜBERLAGERN von Violin und Strip
sns.violinplot(y=x,inner=None,color='lightgray')
sns.stripplot(y=x,jitter=True)
plt.show()




#### LINEARE BEZIEHUNGEN
# regplot()    => einfacher Scatterplot mit Regressionsgeratden
# residplot()  => Residuen
# lmplot()     => kombiniert regplot() mit FacetGrid

## Regplot
diamonds = diamonds.sample(200)
sns.regplot(x='carat', y='price', data=diamonds)
sns.regplot(x='carat', y='price', data=diamonds,\
            order=2, ci=None, scatter_kws={"s": 80},\
            robust=False, lowess=False)
sns.regplot('a','b',df, scatter=True,
                  ax=ax,
                  scatter_kws={'c':df['c']


## Residplot
sns.residplot(x='carat', y='price', data=diamonds)


## lmplot
# => erlaubt: hue, col, row!!!!
# kontinuierliche Werte gehen nicht für hue
diamonds.head(1)
diamonds.nunique()
sns.lmplot(x='carat', y='price', data=diamonds)
sns.lmplot(x='carat', y='price', col='cut', col_wrap=2, data=diamonds)
sns.lmplot(x='carat', y='price',\
           hue='clarity', col='cut', row='clarity',\
           data=diamonds)
sns.lmplot(x='cyl', y='hp', x_estimator=np.mean, data=mtcars);

from ggplot import *
ggplot(diamonds, aes(x='carat', y='price',color='depth')) + \
    scale_color_gradient(low = 'red', high = 'blue') + \
    geom_point() + \
    ggtitle('Test')


##### BI- & MULTIVARIATE  Verteilungen
# 1. 2D Histogramme
# 2. Joint Plots
# 3. Pair Plots
# 4. Heat Maps
dist = np.random.multivariate_normal([-0.5, -0.5], [[1, 0],[0, 1]], 1000).transpose()

## 2D-Histogram
# Bei 2D - auch rechteckige Bins und Hexagons möglich
plt.clf()
_ = plt.hist2d(dist[0],dist[1],bins=(30,30)) # x,y sind 1D arrays
# _ = plt.hexbin(x,x,bins=(10,20)) # x,y sind 1D arrays
plt.colorbar()
plt.show()

## Joint / Scatter Plot
sns.jointplot(x=dist[0],y=dist[1])
plt.show()

## Smoothing
sns.jointplot(x=dist[0],y=dist[1],kind='kde') # kde,scatter,regression,residual,hexbin
plt.show()

## Pairplot
sns.pairplot(df) # hue = categorial
sns.pairplot(df,kind='reg',hue='cat_var')

## Heatmap & Korrelationsmatrix
# wie pseudocolor-plot mit pandas-funktionen
# Zeigt gut die Covariance  => zuerst covariance matrix erstellen - beschreibt wie zwei Variablen sich gleichzeitig ändern
sns.heatmap(covariance,annt=True,  cmap='RdYlGn')  # geht auch mit person-
sns.heatmap(mtcars.corr(), square=True, cmap='RdYlGn') # Gutes Farbschema


#### Lineare Regression

## Einfache lineare Regression
# Kategorische Variablen
# 1. Mit hue='sex'
# 2  Mit col='sex' / row='sex' für subplot
tips = sns.load_dataset('tips')
sns.lmplot(x='tatal bill',y='tip',data=tips, hue='sex',palette='Set1') # Lineare Regressionslinie
plt.show() # => Fächer ~ 95% Konfidenz

## Residuen Plotten
sns.residplot(x='hp', y='mpg', data=auto, color='green')
auto

## Polynomielle Regression
# Erste Ordnung
sns.regplot(x='weight', y='mpg', data=auto, scatter=None, color='blue', label='order 1')
# Zweite Ordnung
sns.regplot(x='weight', y='mpg', data=auto, scatter=None, color='green', label='order 2',order=2)
