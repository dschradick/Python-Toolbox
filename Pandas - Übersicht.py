 ########## PANDAS - ÜBERSICHT

######## GLOBALS IMPORTS & SETTINGS #########
#### Data Manipulation
import pandas as pd
import numpy as np
from datetime import datetime
#### Database
import psycopg2
#### Plotting
import matplotlib.pyplot as plt
import seaborn as sns
#### Linear Modeling
from statsmodels.formula.api import ols, logit
import statsmodels.api as sm
#### TSA Modeling
import statsmodels.tsa.api as smt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
#### Options
import warnings
warnings.simplefilter(action='ignore')
## Pandas
pd.set_option('display.min_rows', 25)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', lambda x: '%.3f' % x) ## keine wissenschaftliche notation
pd.set_option('precision', 3) # nachkommastellen
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 40) # Wieviele Zeichen pro Zelle!!
#pd.set_option('max_colwidth', -1) # kein begrenzung
np.set_printoptions(precision=3, suppress=True)  
## Plotting        
plt.style.use('seaborn-whitegrid')
plt.rcParams['patch.edgecolor'] = 'none'
plt.rcParams['figure.figsize'] = (8, 4)
plt.rc("axes.spines", top=False, right=False)  # despine() als default
## Colours
#https://seaborn.pydata.org/tutorial/color_palettes.html
distinct = sns.color_palette()
sequential = 'Blues'
divergent =  sns.color_palette("RdBu", 10)  # coolwarm, viridis
sns.set_palette(distinct)
#### Default Datasets
mtcars = pd.read_csv('~/Documents/Data/mtcars.csv', index_col='name')
mtcars = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data
diamonds = pd.read_csv('~/Documents/Data/diamonds.csv')
credit = pd.read_csv('~/Documents/Data/german_credit.csv')
gapminder = pd.read_csv('~/Documents/Data/gapminder.csv')
wine = pd.read_csv('~/Documents/Data/wineQualityReds.csv')
online = pd.read_csv('~/Documents/Data/online_clean.csv', parse_dates=['invoice_date'])
airpassengers = pd.read_csv('~/Documents/Data/airline-passengers.csv', parse_dates=['month'], index_col='month')
######## END GLOBALS IMPORTS & SETTINGS

# Seaborn in Pipe
 .pipe(lambda x: sns.barplot(data=x, x='purpose', y='value', hue='variable', palette='Blues_r'))




# Other data
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('~/Documents/Data/ml-1m/movies.dat', sep='::', header=None, names=mnames)
size_in_mb = round(diamonds.memory_usage(deep=True,index=True).sum()/1024/1024,2) # Größe im Speicher

# Vorbereitungen: modellieren 
# besser: create_categories(df), get_dot(df,target) von unten
# => darf NICHT category sein - sondern INT!!!!
credit['creditability'] = credit['creditability'].astype(int) 



## Model
result = logit('am ~ hp', mtcars).fit();              #  am muss INT sein (nicht category/object) => .astype(int) 
result = ols('mpg ~ hp + cyl + am', mtcars).fit()
result.summary()

# Auswertung
anova_table = sm.stats.anova_lm(result, typ=1)
anova_table['explained_variance'] = np.round(anova_table.sum_sq / np.sum(anova_table.sum_sq),2)
anova_table


### Table JOINS
left = pd.DataFrame({'key': ['foo', 'bar','dar'], 'lval': [1, 2, 7]})
right = pd.DataFrame({'key': ['foo', 'bar','mar'], 'rval': [4, 5, 8]})
left.merge(right) # natural inner join
left.merge(right, how='left') 
left.merge(right, how='left', on='key')
left.merge(right, how='left', left_on='key', right_on='key')
left.merge(right, how='left', left_on='key', right_index=True)  # auf index von rechte tabelle für join verwenden
# auf index
left.set_index("key", drop=True, inplace=True)
pd.merge(left,right, how='inner', left_index=True, right_on='key')
# mittels join()  => über den index
right.set_index("key", drop=True, inplace=True)
left.join(right, how='inner')
## Modfied Join-Key - eg. self-join
mtcars.merge(mtcars, left_on=['cyl'], right_on=[mtcars.cyl-2])


### Tests &  Effektstärke
from scipy.stats import norm, ttest_ind, chi2_contingency, t
import scipy.stats as stats
## t-test
np.random.seed(seed=0)
effect = 2
A = norm.rvs(size=1000,loc=2,scale=10)
B = norm.rvs(size=1000,loc=2,scale=10)
result = ttest_ind(A,B+effect)
result.pvalue
conf_low, conf_high = stats.t.interval(0.95, df = len(A)-1, loc = np.mean(A), scale= stats.sem(A))
## chi2-test
group1 = [[u'desktop', 14452], [u'mobile', 4073], [u'tablet', 4287]]
group2 = [[u'desktop', 30864], [u'mobile', 11439], [u'tablet', 9887]]
obs = np.array([[14452, 4073, 4287], [30864, 11439, 9887]])
chi2, p, dof, expected = stats.chi2_contingency(obs)
print(p)

### df(.col).mask(condition, then) vs np.where(condition, then, else) vs df.where(condition, else)
# df.mask():  if-then idiom
# df.where(): if-else idiom
# np.where(): if-then-else idiom


# update mtcars where cyl = 4 set hp=hp+1000

# => df.col.mask(condition, other=value_if_cond_is_true, inplace=True)
mt = mtcars.copy()
mt['hp'] = np.where(mt.cyl==4, mt.hp+1000, mt.hp)
# update mtcars where cyl = 4 and am = 1 set hp=hp+1000
mt['hp'] = np.where(np.logical_and(mt.cyl==4, mt.am == 1), mt.hp+1000, mt.hp)


### Series Erzeugen
index = pd.date_range('2020-01-01',periods=100)
pd.Series(range(0,100), index=index)


#### EINLESEN
!cd ..
!pwd
file_name = '../../../Documents/Data/german_credit.csv'
open(file_name).readline() # => struktur angucken
csv_filepath = '~/Documents/Data/german_credit_test.csv'
col_names = ['col_1','col_2','col_3']
csv_data = pd.read_csv(csv_filepath, index_col=0, names=None, na_values = [0,-1], sep=',',nrows=None)
#xls_filepath = 'Documents/Data/SuperstoreSample.xls'
#xls_data = pd.read_excel(xls_filepath,'Orders')


#### SEPERATING NUMERICAL AND CATEGORIAL
custid = ['customerID']; target = ['Churn']
categorical = data.nunique()[data.nunique()<10].keys().tolist()
categorical.remove(target[0])
numerical = [col for col in data.columns
             if col not in custid+target+categorical]

### DUMMIES ERZEUGEN
# BINARY & ORDINAL
categorials = ['Type of apartment','Occupation','Foreign Worker','Sex & Marital Status','Purpose']
X[categorials] = X[categorials].astype('category')
X = pd.get_dummies(X, drop_first=True)
 # => konvertiert alle object & categorials - andere bleiben wie sie sind
## ORDINAL  => INTEGER
## NOMINAL  => INTEGER
pd.factorize(diamonds.color)  # kann keine order angeben
mapping = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8, 'I':9, 'J':10}
diamonds.color.replace(mapping)

### KATEGORISCHE VARIABLEN
mtcars.cyl.astype('object')  # alte version: string
## Erzeugen
# Standard Weg: nach kategorisch konvertieren
mtcars['cyl'] = pd.Categorical(mtcars.cyl, categories=[4,6,8], ordered=True)      # STANDARD!!
# Alternativ: 
mtcars.cyl.astype('category')
mtcars.cyl.astype('category').cat.as_ordered()
## Typische Funktionen
# Sortierung der Kategorien
mtcars.cyl.cat.reorder_categories([8,6,4], ordered=True);
# Namen ändern
mtcars.cyl.cat.categories = ['Low','Med','High']
mtcars.cyl.cat.rename_categories({4: "Low", 6: "Med", 8: "High"})
mtcars.cyl.cat.categories = [4,6,8]
# Anderes
mtcars.cyl.mode()
mtcars.cyl.min()
mtcars.cyl.value_counts()
# Nach int
mtcars.cyl.astype(int); 


#### PANDAS VERSION
pd._version.get_versions()

#### CHAINING
(mtcars
    .query('hp > 100')
    .groupby('cyl')
    .mean())

mtcars.groupby('cyl',).transform(lambda x: (x - x.mean()) / x.std())

#### DIVERSES
## Auswählen von Spalten basierend auf Typ
mtcars.select_dtypes(include=['number']);              # 'object','datetime'
# Spaltennamen setzen
names = list(mtcars.columns); mtcars.columns = names
## Index aus Spalte generien
mtcars.set_index("name", drop=False, inplace=True)


#### Spaltennamen bereinigen
# BESSER: pyjanitor clean_names()
# Alle gleichzeitig
mtcars.columns = mtcars.columns.str.lower()
mtcars.columns = mtcars.columns.str.replace('\ |\&|\/',"_")
mtcars.columns = mtcars.columns.str.replace('\(|\)',"")
# einzeln
mtcars.columns = mtcars.columns.str.replace(' ', '_')  # Kein "_" => sonst Probleme mit seaborn
mtcars.columns = list(map(str.lower, mtcars.columns))  # Spaltennamen zu lower-case
# Flattening von hierarchischen Spalten-index (z.B. nach mehrfach-aggregagtion)
df.columns = ['_'.join(col).strip() for col in df.columns.values]
df.columns = [col[1].strip() for col in df.columns.values]  # oberes level entfernen
df.columns =  df.columns.droplevel(level=0)



### String 
## Match, Replace & Extract
# Father of the Bride Part II (1995) => jahreszahl
movies.title.str.match(r".*(\d{4}).*")       # beinhaltet jahreszahl?
movies.title.str.replace(r"(\d{4})","1000")  # ersetzen im string
movies.title.str.extract(r"(\d{4})")         # erzeugt neue spalte mit 1995
# zwei matchgroups + extract 
mtcars.index.str.extract(r"([az]+).*(RX)", expand=True)
# Animation|Children's|Comedy
movies.genres.str.split("|", expand=True)    # erzeugt spalten 0,1,2 für die positionen
movies.title.str.isnumeric()                 # type string, aber enthält nur zahlen?
movies.title.astype("object")

# Mapping
a = [1,2,3,4,5]
list(map(lambda x: x*2,a))



#### VISUALSIEREN

# Titel + Achsenbeschriftung
ax = (diamonds.carat.plot(
    kind='hist',
    figsize=(8,6),
    xticks=range(1,4),
    title='TITEL'.upper()))
ax.set_xlabel("x label (unit)");
ax.set_ylabel("y label (unit)")
#---
plt.figure(figsize=(10,10))
sns.heatmap(mtcars.corr(),annot=True,cmap='Blues')    # Korrelationsmatrix
mtcars.hist(bins=10, figsize=(10,10))                 # Histomgram für alle Spaltennamen
mtcars.plot.scatter(x='disp',y='hp',c='qsec')         # Scatterplot mit colorbar
mtcars.groupby('cyl').hist()                          # Histogram für alle Spalten nach Gruppe
mtcars.groupby('cyl').hp.hist(alpha=0.4)              # Überlagertes(!) Histogram für Gruppen
mtcars.groupby('cyl')['hp'].mean().plot(kind='barh')  # !!!! Mittelwerte von Gruppen bzgl. Variable
mtcars[['hp','qsec']]\
    .plot(kind='box', subplots=True, layout=(1,2))    # Boxplots von mehreren Spalten
# Zählen von Werten und Kategorien
mtcars.nunique().plot(kind='barh')                    # Im DF: Anzahl verschiedene distinct Werte
mtcars.cyl.nunique()                                  # Wieviele levels = distinct Werte
mtcars.cyl.value_counts()                             # Anzahl der Vorkommnisse von jedem distinct wert
mtcars.cyl.value_counts(normalize=True)               # Prozent von jedem distinct wert
mtcars.cyl.value_counts().plot(kind='barh')           # In Spalte: Häufigkeit der einzelnen distinct Werte
mtcars.cyl.value_counts().sort_values().plot(kind='barh')   # ....sortiert
# Area => auf Reihenfolge achten - erste ist x!
mtcars.groupby(['cyl','am']).size().unstack().plot(kind='area',stacked=True)
# Legende nach rechts aussen
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
## Tabelle zwischen Textblöcke
from IPython.display import display, HTML
print("Text1")
display(HTML(mtcars.to_html()))
print("Text2")


## DENSITY PLOTS
# !!!! _order macht IMMER auch AUSWAHL (und lässt die anderen faktor-level weg) !!!!
# ECDF => kind="ecdf", complementary=True
## 1 Variable (aufgesplitted nach kategorischer Variabel)
sns.displot(diamonds, x="price", kind="kde", fill=True)
sns.displot(diamonds, x="price", kind="ecdf", fill=True)
sns.displot(diamonds, x="price", hue="color", hue_order=['D','E','F','G','H'], kind="kde", fill=True) # alternativ: multiple='stack'
sns.displot(diamonds, x="price", col="color", col_order=['D','E','F','G','H'], kind="kde", fill=True)
sns.set_theme(style="whitegrid")
sns.displot(diamonds,x="price", hue="color", hue_order=['D','E','F','G','H'], kind="kde", height=6, multiple="fill", clip=(0, None), palette="viridis")
# => zeigt gut die relative Verteilung (unter den Gruppen) in den Tails
# Zwei kontinuierliche Variablen
sns.displot(diamonds, x="price", y='depth', hue="color", hue_order=['D','E','F'])# cbar=True,thresh=.2, binwidth=(10, 1))
# Zwei diskrete Variablen => crosstabluation
sns.displot(diamonds, x="color", y="clarity", aspect=1.2, cbar=True)
# JointPlot + Boxplots für die Randverteilungen
g = sns.JointGrid(data=diamonds, x="price", y="depth")
g.plot_joint(sns.histplot)
g.plot_marginals(sns.boxplot)

## FACETGRID
g = sns.FacetGrid(data=(diamonds
                        .groupby(['cut','color'], group_keys=False)
                        .sample(200, replace=True)),        # => Stratified (Disproportionale) Sample
                  row="cut", col="color", #hue='clarity', 
                  row_order=['Fair','Good','Very Good'],    # => !! Weglassen von Werten = weglassen von Graphen
                  col_order=['D','E','F']) # 
g.map(sns.regplot, 'carat', 'price', 
      x_estimator=np.median, x_bins = 30,                   # x_bins = bins aus x var generieren
      ci=False, lowess=True)                                # lowess GEHT bei regplot
#g.map(sns.scatterplot, 'carat', 'price')


## 3 kontinuierliche Variablen
df = (diamonds
      .groupby([pd.cut(diamonds.depth,20),  # Y
                pd.cut(diamonds.carat,20)])  # X
              ['price']                    # Ziel Variable
      .max() # Statistik: count, sum, mean, median, max, min,...
              # => rumspielen
      .unstack()
      .sort_index(ascending=False))

# Heatmap: 'coolwarm' => von minus bis plus (z.B. Korrelation)
#          'Blues' => wenn nur aufsteigend (z.B. counts)
sns.heatmap(df, annot=False, fmt=".0f", linewidths=.05, cmap='Blues');

## Correlation-Heatmap
corr = mtcars.corr(method = 'pearson')
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(11, 9))       
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, 
            mask = mask, cmap = cmap, 
            vmax = 1, vmin = -1,                                     
            center = 0, square = True,                                 
            linewidths = 1.5, cbar_kws = {"shrink": .9},                     
            annot = True);
plt.xticks(rotation=45)                                   
plt.yticks(rotation=45)                                   
plt.title('Diagonal Correlation Plot', size=30, y=1.05);  

### Contingency-tables
table = pd.crosstab(index=mtcars.am,
                    columns=[mtcars.cyl,mtcars.gear],
                    margins=True)
table              # crosstable
table.iloc[:,3]    # Spalte

## Untergruppen(Anzahl): Aggregation bzgl 2 Variablen
pd.crosstab(mtcars.cyl,mtcars.carb).plot(kind='barh', stacked=True, cmap='viridis')
# Beliebige Aggregation 2er Variablen => eine Targetvariable
# 1. Barchart oder 2. Heatmap
# => gut bei ordinalen Variablen
# => visualisieren von Interaktionseffekten
plt.figure(figsize=(14,10))
cuts=12
ax = (sns.heatmap(
        wine.groupby([
            pd.cut(wine['volatile_acidity'], cuts),       # Y
            pd.cut(wine['alcohol'],cuts)])                # X
        ['quality']
        .mean()
        .unstack(),
        center=7,                                         # WICHTIG - gut zum hervorheben!!!
        linewidths=.5, fmt=".1f", cmap='viridis_r',
        annot=True,  cbar=True))

### Log
## Schnell
plt.xticks([0,1,2,3,4],[1,10,100,1000], rotation=45)
## Aufwendig
base   = 10
ticks  = np.arange(1,4)
labels = np.round([base ** tick for tick in ticks])
plt.xticks(ticks, labels, rotation=45)





#### DATEN BESCHREIBEN
mtcars.head(5).T                               # => gut wenn viele Spalten!
mtcars.describe().transpose()
mtcars.shape; mtcars.dtypes, mtcars.nunique()
mtcars.info(); mtcars.describe()               # Allgemein
## Distinkte Werte und Werte zählen
mtcars.nunique() # auf df!!                    # ANZAHL der (distinct) Werte in JEDER SPALTE
# - Kategorische
mtcars.cyl.unique()                            # WELCHE distinkten Werte gibt es
mtcars.groupby('cyl').size()                   # anzahl der unique werte
mtcars.cyl.value_counts()                      # ANZ. der WERTE IN LEVEL(!) in Kategorien
mtcars.cyl.max(); mtcars.cyl.idxmax()          # Max-Element / index von Max-Element
mtcars.quantile([0.25,0.75])                   # Quantile

# -
mtcars.groupby('cyl').size().sort_values()     # =...
freq_table = pd.crosstab(mtcars.cyl,mtcars.carb)
freq_table
sns.heatmap(freq_table,annot=True, fmt="d", linewidths=.05, cmap='Blues')
# oder
 # => beide mit plot(kind='barh')


### KORRELATION
# Kurz
sns.heatmap(mtcars.corr(),annot=True)          
# Lang
mask = np.triu(np.ones_like(mtcars.corr(), dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
plt.figure(figsize=(7,7))
sns.heatmap(mtcars.corr(), vmin=-1, vmax=1, center=0, 
            cmap=cmap,fmt='.2f',annot=True, 
            linewidths=0.5, mask=mask, square=True)
plt.xticks(rotation=40)


### Sortieren
mtcars.sort_values('hp',ascending=True)
mtcars.sort_values(['hp', 'mpg'], ascending=True, inplace=False)

## sql-distinct
mtcars.drop_duplicates()

#### BEOBACHTUNGEN
### AUSWAHL
airpassengers.loc['1950':'1951-02']            # für DatetimeIndex
mtcars.head(1); mtcars.tail(1)                 # Anfang / Ende von df
mtcars.query('carb > 4')                       # Filter  # => nicht performant - besser: ..
mtcars[mtcars.carb > 4]                        # Einfache Auswahl
mtcars[(mtcars.carb > 4) & (mtcars.cyl > 6)]   # Auswahl mit Und
mtcars[mtcars.cyl.isin([4,6])]                 # IN => wichtig für joins
mtcars.nlargest(2, 'hp')                       # Reihen mit größtem Element bzgl. Spalte
# Über Position
mtcars.iloc[1:10]                              # Position
mtcars.loc['Mazda RX4':'Datsun 710']           # Reihenname
mtcars[mtcars.carb > 4][['cyl','mpg']]         # Filter + Spaltenauswahl
mtcars.loc[mtcars.carb > 4,['cyl','mpg']]      # =...
mtcars.query('carb > 4 & hp > 150')            # Fiter mehrerer Variablen
mtcars[(mtcars.carb > 4) & (mtcars.hp > 150)]  # =...
mtcars[mtcars.name.str.startswith('Maz')]
mtcars[mtcars.name.str.contains('Maz')]        # Filter nach String
mtcars[~mtcars.name.str.contains('Maz')]       # Filter nach nicht(!) spezieller String
np.where(mtcars.mpg > 20)                      # Indizes, welche Bed. erfüllen
mtcars.duplicated(subset=['cyl'],keep='first') # Finde duplicates - keep: first, last, False = 
mtcars.duplicated().any()
mtcars.cyl.drop_duplicates(ignore_index=True)  # distinct - keep, first, last, False = Drop all
mtcars.cyl.unique()                            # distinct für kategorisch
mtcars.cyl.sample(n=10)                        # sample
mtcars.cyl.sample(frac=0.1)                    # sample
### UPDATE
mtcars.loc[mtcars.hp < 100,'hp'] = 100         # wird inplace gemacht



#### SPALTEN
### Auswahl
## Einfach
mtcars.hp                                      # ein Spalte nach Name
mtcars['hp']                                   # =...
mtcars[['hp', 'cyl']]                          # Mehrere Spalten
mtcars[mtcars.columns[1]]                      # Spaltennummer
mtcars.iloc[:,:5]                              # Erste 5 Spalten
## Komplexere Auswahl
selection = mtcars.columns.str.contains('cyl')
mtcars[mtcars.columns[selection]]              # Filter auf Namen - Alternativ..
mtcars.filter(regex='^cy',axis=1)              # Regex auf Spaltenname
#mtcars.select_columns(['cyl','carb','disp'],invert=True)# ALLE SPALTEN AUSSER
mtcars.select_dtypes(include=['object'])       # 'number','object','datetime'
### Löschen
del mtcars['name']                             # Spalte löschen
### Hinzufügen
# Basierend auf einer Spalte
small_vs_large = lambda x: 'large' if x > 20 else 'small'
mtcars.mpg.apply(small_vs_large)    
# Basierend auf mehreren Spalten
high_hp_am = lambda x: 'yes' if x.hp > 100 and x.am == 1 else 'no'
mtcars['high_hp_am'] = mtcars.apply(high_hp_am, axis=1)
## np.where vs col.where vs col.mask 
np.where(mtcars.hp > 100, 'high','low')        # np.where() => if-then-else   (aber kein index)
mtcars.hp.mask(mtcars.hp > 100, 'high hp')     # series/df.mask()  => if-then 
mtcars.hp.mask(lambda x: x > 100, 'high hp')   # =
mtcars.hp.where(mtcars.hp > 100, 'low hp')     # series/df.where() => if-else
mtcars['new_col'] = range(0,32)                # Werte
mtcars['new_col'] = mtcars.hp - mtcars.mpg     # auf Basis anderer Spalten
mtcars.assign(new_col=mtcars.hp - mtcars.mpg)  # =... => für chaining
 # => assign für chaining (wie mutate)
### Umbenennen
mtcars.columns                                 # ist index (beinhaltet .str funktionen)
mtcars.rename(columns={'hp': 'ps'})            # Umbenennen



#### REIHEN
mtcars.head(1)
mtcars['name'] = mtcars.index                  # Index => Spalte
mtcars.set_index('name')                       # Spalte => index
mtcars = mtcars.reset_index(drop=True)         # Erzeugt normalen numerischen Index #  # (drop = True =>  alter index nicht als Spalte anlegen)
mtcars.index = mtcars.name                     # alternativ



#### Reihe + Spalte
# [Reihe,Spalte]
mtcars.iloc[1:10,:]                                            # Numerisch: [Reihe,Spalte]
mtcars.iloc[:,1:10]
mtcars.iloc[1:10,1:3]
mtcars.loc['Mazda RX4':'Datsun 710',['hp', 'cyl']]             # Nach Namen
# mtcars.iloc[0:10,'hp'] geht nicht # => kein mixing
## Mischen von integer und string adressierung
mtcars.ix[0:10,'hp'] # => geht, aber deprecated                # Numerisch + Namen mischen => deprecated
# Statt .ix - besser:
mtcars.iloc[0:10, mtcars.columns.get_loc('hp')]                # ix für eine column
mtcars.iloc[0:10, mtcars.columns.get_indexer(['hp', 'cyl'])]   # ix für mehrere columns
mtcars.loc[mtcars.index[0:10], 'hp']                           # ...alternativ

## Summary stats
mtcars.describe()                              # Dataframe
mtcars.mpg.describe()                          # Spalte
mtcars.query('mpg > 20').describe()            # Subset


### TRANSFORMATION
from scipy.stats import boxcox
def boxcox_transform(x): return boxcox(x)[0]
mtcars[['hp']].apply(boxcox_transform, axis=0)     # DataFrame  => .apply()
mtcars.hp.transform(boxcox_transform)              # Series     => .transform()

## Gruppierung
grouping = mtcars.groupby('cyl')                    # DataFrameGroupBy object
grouping.groups                                     # dict
grouping.first()                                    # erste row aus jeder gruppe
grouping.indices.keys()                             # dict_keys([4, 6, 8])
grouping.get_group(6)                               # gruppe mit dict-key 6
# Aggregieren
grouping.mean()
grouping['hp'].mean()
# Funktion anwenden
def top5(group_df, sort_col='hp'): 
    return group_df.sort_values(sort_col).head(5)
grouping.apply(top5)
grouping.apply(top5, 'disp')   #


#### AGGREGATION
## Anzahl & Proportion
mtcars.cyl.count()                             # Anzahl der Einträge
mtcars.cyl.value_counts()                      # Anzahl Einträge nach Faktor
np.sum(mtcars.hp > 150)                        # Anzahl mit Bedingung
np.mean(mtcars.hp > 150)                       # Proportion
# named aggregation: 
#  select mean(CustomerId) as customer_cnt
mtcars.groupby('cyl').agg(mean_hp=('hp','mean'))
## Gruppierung
mtcars.groupby('cyl').mean()                   # auf ganzen dataframe berechnen
mtcars.groupby('cyl')['hp'].count()            # nur auf spalte
mtcars.groupby('cyl').agg({'hp': 'mean'})      # = ...
mtcars.groupby('cyl')['hp'].count().plot(kind='bar')
# Grouper (TimeGrouper gibts nicht mehr)
online.groupby(online.invoice_date.dt.year).mean() # =...
# origin = timestamp on which to adjust the grouping 
#        = {‘epoch’, ‘start’, ‘start_day’}
# offset = offset timedelta added to the origin
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Grouper.html?highlight=grouper#pandas.Grouper
online.groupby(pd.Grouper(key='invoice_date', freq="1Y")).mean() 
## HAVING
# (nicht spalte nach groupby angeben(!!!)
# having count(*) > 12
(mtcars
    .groupby('cyl')
    .filter(lambda x: len(x) > 12)
# select sum(mpg) from mtcars where am = 1 group by cyl having sum(mpg) > 100
(mtcars
    .query('am == 0')                         # WHERE
    .groupby('cyl')                           # GROUP BY für HAVING
    .filter(lambda x: x['mpg'].sum() > 100)   # HAVING-Bedingung
       # => liefert subset von originalen df mit den original Werten!! (nicht gruppiert)
    .groupby('cyl')['mpg']                    # Eigentliches GROUP BY
    .sum())                                   # Aggregation für Gruppe
## Komplexe Grupperierung / Aggregationen
# Eine Spalte mit mehreren Funktionen
mtcars.groupby('cyl').agg({'hp':[min,max,'mean']})  # mehrere funktionen
##  Mehrere Spalten mit mehreren Funktionen
aggregations = {
    'hp' : [min,max,'mean'],
    'mpg': [min,max,'mean']}
mtcars.groupby.__doc__
df = mtcars.groupby('cyl').agg(aggregations); df
# => nach aggregation multiindex in spalten entfernen
df.columns = ['_'.join(col).strip() for col in df.columns.values]; df
## Umbenennen
# z.B. hp-unterlevel-min  ==> hp_min
grouped = mtcars.groupby('cyl').agg({'hp':[min,max,'mean']}); grouped
grouped.columns = ["_".join(x) for x in grouped.columns.ravel()]; grouped
## Transform -
# fügt Wert in die entsprechenden Werte in Zellen einer kopie des original-df ein
mtcars.assign(mean_hp_in_cyl=mtcars.groupby('cyl')['hp'].transform('mean'))
## Levels - Gruppen nach Level im MultiIndex
mtcars2 = mtcars.set_index(['cyl'], append=True)    # Hinzufügen von cyl zu index
mtcars2.groupby(level=1).sum()






#### CLEANING
# #Strings aufräumen
diamonds.cut.str.replace('Fair','Ok')              # STRINGFUNKTIONEN
replacement = {                                    # mittels dict
    "Ideal": "OK",
    "Ok": "OK",
    "HELLO": "CORP"
}
diamonds.cut.replace(replacement)                  # KEIN str!
pattern = "(Ideal|Ok|Normal)"                      # Regex
diamonds.cut.str.replace(pattern, 'OK', regex=True)

## Einträge löschen - (generelles slicen)
mask = diamonds.cut.str.contains('Good')
mask = ~diamonds.cut.str.endswith('Good')
diamonds[mask]
## String-column aufsplitten
new = diamonds.cut.str.split(' ', n = 1, expand = True)
diamonds["First"] = new[0]
diamonds["Second"] = new[1]

### Type-Casting
mtcars['cyl'] = mtcars.cyl.astype("object")
# Wenn voher object => pd.to_numeric
mtcars['cyl'] = pd.to_numeric(mtcars.cyl, errors='coerce')
# Wenn voher schon number => z.B.  int8|16|32|64,float64,boolean
mtcars['cyl'] = mtcars.cyl.astype('int8')

### Duplikate
mtcars.drop_duplicates()                            # Duplikate entfernen
mtcars.drop_duplicates(subset='cyl', keep='first')  # Duplikate entfernen (wenn duplikat auf id)

### Missing Values
# df.any() = gibt es ein element in df/series, welches true ist?
# df.all() = sind alle elemente in df/series true?
# isna() == isnull()
mtcars.isna().any()                                # In welcher Spalte gibt es nans?
mtcars.isna().any().any()                          # Gibt es nans in df?
mtcars.isna().sum()  #!!!                          # Wieviele NANs in welcher Spalte?
mtcars.isna().sum().sum()                          # Wieviele NANs in df?
## Zeig die Reihen
mtcars.loc[mtcars.T.isnull().any()]                # Alle Reihen mit NANs
mtcars.loc[mtcars.cyl.isnull()]                    # Reihen mit NANs in Carat

### Nans behandeln
mtcars.dropna(axis=0, how='any')                   # Droppen: wo "any" or "all" der Daten Missing
mtcars.dropna(subset=['name', 'cyl'])              # Droppen: wenn spezielle Spalten missing haben
mtcars.cyl.fillna(mtcars.cyl.mean())               # Ersetzen von NaN
mtcars.cyl.replace(to_replace=np.nan, value=10)    # Ersetzen: von NaN durch 10
mtcars.cyl.replace(to_replace=-1, value=10)        # Ersetzen: von 4 durch 10
mtcars.loc[mtcars.hp < 100,'hp'] = 100             # ganzes subset

### Als dict zurückgeben
mtcars.set_index('name').hp


#### FREQUENZ-TABELLE
ft = mtcars.cyl.value_counts(); ft                                    # Frequency-Table
ft / ft.sum()                                                         # Frequency-Table Proportionen
pd.crosstab(mtcars.cyl, mtcars.carb, margins=True)                      # Anzahl
    pd.crosstab(mtcars.cyl, mtcars.carb, margins=True, normalize='index')   # Proportion
pd.crosstab(mtcars.cyl, mtcars.carb, margins=True).plot(kind='barh')


## Pivot Table
# Wenn man zwei kategorische / ordinale Variablen hat die bzgl. einer andern aggregiert werden sollen
# Inbesondere gut für Jahr & Monat => avg(Preis)
pd.pivot_table(mtcars,   columns='am',    index='cyl',          values='hp',    aggfunc='mean') # count
pd.pivot_table(mtcars,   columns='am',    index=['cyl','gear'], values='hp',    aggfunc='count', fill_value=0)
pd.pivot_table(diamonds, columns='color', index='cut',          values='price', aggfunc=np.mean, margins=True)

## Melt
# id_vars    = schlüssel variablen => bleiben als einzige zusätzliche Spalte vorhanden (+ val_name_col & value_col)
# value_vars = spalten welche gunpivoted werden sollen 
# => !!andere spalten welche nicht in id_vars oder value_vars sind werden verworfen!!
df = pd.pivot_table(diamonds, columns='color', index='cut', values='price', aggfunc=np.mean)
df.melt(id_vars=['D','E'], value_vars=['F','G'], 
        var_name='Color', value_name='mean_price')
#         D        E Color  mean_price
# 3631.293 3538.914     F    4324.890
# 3470.467 3214.652     F    3778.820
# ...
# 4291.061 3682.312     G    4239.255
# 3405.382 3423.644     G    4123.482
# => !!!!  Inforamation von H,I Spalten verworfen, weil weder in id_vars noch value_vars sind!!!!


#### DISKRETISIEREN
## Binäre Variable erzeugen
# 1. Vektor-Version
mtcars['binary_variable'] = 1 if mtcars.hp > 5 else 0
# 2. Reihenweise
## A. Einfach
mtcars['binary_category'] = mtcars.hp.apply(lambda x: 0 if x <= 120 else 1)
## B. Funktion
def discretize_target(x):
    if x < 70:
        return "low"
    elif x < 100:
        return "med"
    else:
        return "high"
mtcars['mpg_cat'] = mtcars.hp.apply(discretize_target)
## Range zerlegen
bins = np.arange(0,5,0.5)         # range nimmt keine floats!
labels=['low', 'medium', 'big']                                  # n labels
bins = [mtcars.hp.min()-1, 100, 200, mtcars.hp.max()+1]          # dafür n+1 grenzen
mtcars['hp_cat'] = pd.cut(mtcars.hp, bins=bins, labels=labels)   # kategorisch erstellen



### APPLY & MAP
square = lambda x: x ** 2
mtcars['squared_hp'] = mtcars.hp.apply(square)
# =>  standard python: list comprehension ist syntactic sugar für map/apply
mtcars['squared_hp'] = list(map(square,mtcars.hp))


#### Reordering von Factors
mtcars['cyl'] = mtcars['cyl'].astype('category')
mtcars['cyl'] = mtcars['cyl'].cat.reorder_categories([4, 6, 8], ordered=True)



#### DATE-TIME
# freq: S = second, 10S = 10 seconds
# H = hour, D = day, B = businessday, W = week, Y = year
# M = month end(!), MS = month start
# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
pd.date_range('2013-12-15', periods=6, freq='MS')
### Konvertieren
mtcars['build_date'] = "2017-01-01 12:10:02"              # Datetime String
mtcars['build_date'] = pd.to_datetime(mtcars.build_date)  # Konvertierung
pd.to_datetime("2017-1-1-15", infer_datetime_format=True)
pd.to_datetime('20141101 1211', format='%Y%m%d %H%M', errors='ignore')
### date_trunc 
## am besten: to_period => period als Type
online.invoice_date.dt.to_period('M')
online.invoice_date.dt.to_period('M').dt.to_timestamp()
## oder als date behalten
get_month = lambda x: datetime(year=x.year, month=x.month, day=1)
online.invoice_date.apply(get_month)
# strftime
 pd.to_datetime(online.invoice_date.dt.strftime('%Y-%m'))

## Zeit zwischen zwei dates
# eg. monate   => hier normalisiert auf monatsanfang
(online.invoice_date.dt.to_period('M') - online.invoice_date.apply(get_month).dt.to_period('M')).apply(lambda x: x.n)
# wie für ab dem tatsächlichem datum?!
.assign(retained_day=lambda x: (x.invoice_date - x.first_day).dt.days,  # bessere möglichkeit?
        retained_month=lambda x: x.retained_day // 30,
## Als index
date_index = pd.date_range(start='2017-01-01', periods=len(mtcars))
mtcars2 = mtcars.set_index(date_index)
mtcars2.loc['2017':'2018']
mtcars2.loc['2017-01-05':'2017-01-10']
## Multiindex
mtcars2['date'] = date_index
mtcars2 = mtcars2.set_index(['cyl','date'], drop=False)
mtcars2['cyl'] = mtcars2.cyl.astype(int)
mtcars2.loc[[4,'2017']]  # anderesrum geht datum-slicen nicht
### Extrahieren
mtcars.build_date.dt.year
mtcars.build_date.dt.minute
mtcars.build_date.dt.strftime('%m/%d/%Y %H:%M')
# %Y Four-digit year %y Two-digit year
# %m Two-digit month [01, 12] %d Two-digit day [01, 31]
# %H Hour (24-hour clock) [00, 23]
# %I Hour (12-hour clock) [01, 12]
# %M Two-digit minute [00, 59]
# %S Second [00, 61] (seconds 60, 61 account for leap seconds) %w Weekday as integer [0 (Sunday), 6]
# %U Week number of the year [00, 53]; Sunday is considered the first day of the week, and days before the first Sunday of the year are “week 0”
# %W Week number of the year [00, 53]; Monday is considered the first day of the week, and days before the first Monday of the year are “week 0”
# %z UTC time zone offset as+HHMMor-HHMM; empty if time zone naive %F Shortcut for%Y-%m-%d(e.g.,2012-4-18)
# %D Shortcut for%m/%d/%y(e.g.,04/18/12)


### Gruppieren
mtcars['build_date'] = pd.date_range(start='2017-01-01', periods=len(mtcars))
mtcars.groupby('build_date').count().plot()
mtcars.groupby(mtcars.build_date.dt.year)['mpg'].count().plot()
### Auswahl
mtcars[mtcars.build_date > '2017-01']
mtcars[mtcars.build_date > '2017-01-30']

## Multiindex
d2 = diamonds.set_index(['color','cut'])
d2.loc[('E',)]           # nicht  d2.loc[(,'Ideal')]
d2.loc[('E','Ideal')]


#### Einlesen von grossen Dateien
# Sonst: MemoryError
chunksize = 10 ** 6      #  chunksize = anzahl der reihen pro chunk
for chunk in pd.read_csv(filename, chunksize=chunksize):
    process(chunk)

#### Nach numpy
df.to_numpy()
df.values

#### Konvertieren
_ = pd.Categorical(mtcars.am, categories=[0,1], ordered=True)
def convert_to_cat(column, ref_level=None):
    order = sorted(column.unique())
    if ref_level:
        order.remove(ref_level)
        order = [ref_level] + order
    column = column.astype('category')
    column = column.cat.reorder_categories(order, ordered=True)
    return column
def create_categories(df, max_uniques=10):
    categorials = list(df.columns[df.nunique() < max_uniques])
    df[categorials] = df[categorials].apply(convert_to_cat)
# 1. General
create_categories(diamonds)
# 2. Refine
diamonds['cut'] = diamonds.cut.cat.reorder_categories(['Fair','Good','Very Good','Premium','Ideal'])


#### Der R-"." für lineare Modellierung
def get_dot(df, target, include_categories=True):
    predictors = df.drop(target, axis=1)
    pred_names = predictors.select_dtypes(exclude='category').columns
    if include_categories:
        cats = predictors.select_dtypes(include='category').columns
        pred_names = list(pred_names.append(cats))
    dot = " + ".join(pred_names)
    return dot


### Factors
def fct_lump(series, n=3, other='Other'):
    others = list(series.value_counts()[n:].index)
    print(len(others))
    if len(others) <= 1:
        return series
    othered = pd.Series(np.where(np.isin(series, others), other, series))
    return othered



############ ANDERES

### select distinct col1, col2,...
mtcars[['cyl','am']].drop_duplicates()  # => gut zum custom_index erstellen

### coalesce
df['D'] = df.D.fillna(df.A).fillna(df.B).fillna(df.C)  # Ordnung nach der gefüllt: A,B,C

### case
mtcars['hp_cat'] = np.where(
     mtcars.hp.between(0, 100, inclusive=False),
    'Small',
     np.where(
         mtcars.hp.between(101, 200, inclusive=False),
         'Medium',
         'Large'))
# oder:
mtcars['hp_cat'] = (
    np.select(
        [mtcars.hp.between(0  , 100, inclusive=False),
         mtcars.hp.between(101, 200, inclusive=True)],
        ['Small',
        'Medium'],
        default='Large'))


### HAVING         => .groupby().filter(lambda group_df: ..)
# having count(1) > 10
mtcars.groupby('cyl').filter(lambda df: len(df) > 10)
# SELECT avg(hp) FROM mtcars group by cyl having count(*) > 10
# => filter() gibt ungruppierten df zurück!!!!
(mtcars
    .groupby('cyl')
    .filter(lambda df: len(df) > 10)
    .groupby('cyl')
    ['hp']
    .mean())
mtcars.filter()
# having sum(hp) < 1000  (=> gibt alle(!) 4 & 6 zylinder als EINEN df)
mtcars.groupby('cyl').filter(lambda df: df.hp.sum() < 1000) # hier: result sind nur die 8-cyl

### in
 mtcars[mtcars.cyl.isin([6,8])]

### between
 mtcars[mtcars.cyl.between(6,8)]

### Self-join mit modifiziertem key
mtcars.merge(mtcars, left_on=['cyl'], right_on=[mtcars.cyl-2])



# Group-apply kann beliebige funktion auf Gruppen-df anwenden und beliebigen df zurückgeben!
# => Rückgabe: beliebiger Dataframe
mtcars.groupby('cyl').apply(lambda x: x)                 # identity        
mtcars.groupby('cyl').apply(lambda x: x[['am','vs']])    # subsetting
mtcars.groupby('cyl').apply(lambda x: x.rename(columns={'am':'bm'})) # umbenennen
mtcars.groupby('cyl').apply(lambda x: x.hp.cumsum())     # akkumulieren
# !!
mtcars.groupby('cyl').apply(
    lambda x: pd.DataFrame(data={'a':x.hp,'b':x.am}))    # komplett(!) neuer Dateframe
# => kann aus Gruppen-DF beliebige neue Dateframe bauen!!!!!
  

#### WINDOW FUNCTIONS
for window in pd.Series(range(10)).rolling(window=3): print(window)
for window in pd.Series(range(3)).expanding(): print(window)
# for window in pd.Series(range(3)).ewm(window=3, alpha=.8): print(window) geht nicht
# Einfache Window Function selber schreiben:
# => müssen länge des der transformierten Spalte zurückgegben
min_transform = lambda x: [np.min(x)] * len(x)
mtcars.hp.transform(min_transform)
z_score = mtcars.hp.transform(lambda x: (x - x.mean()) / x.std())
mtcars.groupby('cyl').hp.transform(np.min)   # geht        => skalar wird gebroadcasted (siehe doc von grouping)
mtcars.hp.transform(np.min)                  # geht NICHT  => nicht automatisch gebroadcasted


# .transform(func) 
# cumsum = expanding.cumsum() bei unbounded preceding
# SELECT cyl,hp,sum(hp) OVER (PARTITION BY cyl ORDER BY hp )
# transform() vs apply()(welches einen wert zurück gibt)
# ==..  analog group by vs over() bei sql
# Jede Reihe wird abgebildet auf den den skalaren Wert den Funktion zurückliefert


##  rank() - ASC & DESC
#  ===> Parameter von rank() - nicht von der sortierung(!!)
#  => dense_rank() over(order by hp desc) == sort_values(hp).rank(ascending=False)
#  mtcars.hp.assign(rnk=lamba x: x.hp.sort_values().rank(ascending=False))

## rank(): TIES
# rank: how it breaks ties / tie values                                 SQL
# - default = average : assign each group the mean rank
# - first             : row number # wenn sort_values() => 1,2,3,4      row_number()
# - dense:            : assign minimum rank - keine Lücken              dense_rank()
# - min:              : assign minimum rank for whole group - Lücken    rank()
# - max:              : assign maximum rank for whole group         
#
# => KLARMACHEN MIT: mtcars.hp.sort_values().rank(method='first', ascending=True)  !!!!!!
#
#    (sort_values() nur für ausgabe - unwichtig für zuweisung)

## Neue Spalte mit mittels window function
mtcars.assign(total_hp=lambda x: x.groupby('cyl').hp.transform('sum'))

def window_rank(group_df): return group_df.rank(method='min')                  # SQL: rank()
def window_dense_rank(group_df): return group_df.rank(method='dense')          # SQL: dense_rank()
def window_row_number(group_df): return group_df.rank(method='first')          # SQL: row_number()
def window_cume_dist(group_df): return group_df.rank(pct=True, method='max'))  # SQL: cume_dist()
def window_ntile_100(group_df): return group_df.rank(pct=True)  # falsch => was ist percent_rank in pandas?
def window_sum(group_df): return pd.Series([group_df.sum()] * group_df.shape[0])
def window_count(group_df): return group_df.count()
mtcars['ntile_100'] = mtcars.groupby('cyl')['hp'].rank(pct=True) # =...
mtcars['ntile_100'] = grouping['hp'].transform(window_ntile_100)
mtcars['row_number'] = mtcars.hp.transform(window_row_number)   # ordering für rank() muss durch auswahl von zeile
mtcars.hp.sort_values().transform(window_ntile_100)
mtcars.hp.sort_values().transform(np.cumsum) # mtcars.hp.cumsum() vs # mtcars.hp.sum() => sum() liefert nur einen wert
mtcars.groupby('cyl').hp.transform(window_sum)
mtcars.cyl.rank(pct=True, method='max').sort_values()

mtcars.rank() # auf df: für jede spalte

mtcars['hp_cumsum'] = (mtcars.sort_values('hp')
                             .groupby('cyl')
                             ['hp']
                             .cumsum());   #  alternativ: .transform(np.cumsum)); .transform(lambda x: x.rank()
mtcars[['cyl','hp','hp_cumsum']].sort_values(['cyl','hp_cumsum'])
# SELECT cyl,hp,sum(hp) OVER (PARTITION BY cyl ORDER BY hp rows between 2 preceding and current)
(mtcars
    .sort_values('hp')
    .groupby('cyl')
    .hp
    .transform(lambda x : x.rolling(3).sum()))
mtcars[['cyl','hp','hp_cumsum']].sort_values(['cyl','hp_cumsum']).head(5)
# select cyl,hp,rank() over (partition by cyl order by hp desc)
mtcars['hp_rank'] = (mtcars.sort_values('hp')
                             .groupby('cyl')['hp']
                             .rank(method='first', ascending=False));
mtcars[['cyl','hp','hp_rank']].sort_values(['cyl','hp_rank']).head(20)
