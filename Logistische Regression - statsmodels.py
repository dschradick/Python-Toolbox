########## LOGISTISCHE REGRESSION - STATSMODELS
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

#### Daten einlesen
df = pd.read_csv("https://stats.idre.ucla.edu/stat/data/binary.csv")
# Spalten umbenennen (wg. "rank" Funktion von pandas dataframe)
df.columns = ["admit", "gre", "gpa", "prestige"]

#### Daten betrachten
df.head(2)
df.describe()
df.std()
pd.crosstab(df['admit'], df['prestige'], rownames=['admit'])
df.hist()
pl.show()


#### Dummies erstellen
dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')
dummy_ranks
cols_to_keep = ['admit', 'gre', 'gpa']
data = df[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])
#data['intercept'] = 1.0
data.head(2)

#### Regression durchführen
train_cols = data.columns[1:]
data['admit']
logit = sm.Logit(data['admit'], data[train_cols])
result = logit.fit()

#### Resultat
## R-like Output für die Regression
result.summary()
# => wahrscheinlichkeit, wenn schule mit höherem prestige besucht (1 = höchstes) angenommen zu werden ist höher
result.conf_int()  # Konfidenz-Intervalle
# e^a = exp(a) - np.exp(1) = 2,7...
np.exp(result.params)  # Nur die Odds-Ratios
# generiert odds-ratio
# => wie anstieg / abfall um eine Einheit einer ggb. variable das odds-ratio angenommen zu werden beeinflusst
# => odds admitted zu werden ist für prestige=2 50% von reference-level prestige=1

## Mit Konfidenz-Intervallen
# => um die Unsicherheit in den Variablen zu quantifizieren
params = result.params
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
np.exp(conf)

a = np.linspace(1,10,10)
plt.plot(a,np.exp(a))

#### AUC, ROC & Cut-off
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

## AUC
prediction = result.predict(data[train_cols])
roc_auc_score(data['admit'], prediction)

## ROC Kurve
fpr,tpr,thresholds = roc_curve(data['admit'],prediction)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC-Curve')
plt.show()

## Cut-Off
# => tpr hoch, fpr low
# tpr - (1-fpr) ist 0 oder nahe 0 ist optimaler cut-off
i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),
         'tpr' : pd.Series(tpr, index = i),
         '1-fpr' : pd.Series(1-fpr, index = i),
         'tf' : pd.Series(tpr - (1-fpr), index = i),
         'thresholds' : pd.Series(thresholds, index = i)})
roc.ix[(roc.tf-0).abs().argsort()[:1]]

fig, ax = plt.subplots()
pl.plot(roc['tpr'])
pl.plot(roc['1-fpr'], color = 'red')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])


#### Weiteres zur Evalulierung
# Datensatz neu erzeugen mit alle logischen Kombinationen der Input-Werte
# => zum betachten wie sich vorhergesagte Wahrscheinlichkeit entlang der Variablen verändert

# Für GRE und GPA nicht alle möglichen Wertenm, sondern 10 Werte von min bis max
gres = np.linspace(data['gre'].min(), data['gre'].max(), 10)
gpas = np.linspace(data['gpa'].min(), data['gpa'].max(), 10)


## Hilfsfunktion
 def cartesian(arrays, out=None):
      arrays = [np.asarray(x) for x in arrays]
      dtype = arrays[0].dtype

      n = np.prod([x.size for x in arrays])
      if out is None:
          out = np.zeros([n, len(arrays)], dtype=dtype)

      m = n / arrays[0].size
      out[:,0] = np.repeat(arrays[0], m)
      if arrays[1:]:
          cartesian(arrays[1:], out=out[0:m,1:])
          for j in xrange(1, arrays[0].size):
              out[j*m:(j+1)*m,1:] = out[0:m,1:]
      return out

# Alle möglichen Kombinationen mittels cartesian()
combos = pd.DataFrame(cartesian([gres, gpas, [1, 2, 3, 4], [1.]]))
# Dummy-Variablen
combos.columns = ['gre', 'gpa', 'prestige', 'intercept']
dummy_ranks = pd.get_dummies(combos['prestige'], prefix='prestige')
dummy_ranks.columns = ['prestige_1', 'prestige_2', 'prestige_3', 'prestige_4']
cols_to_keep = ['gre', 'gpa', 'prestige', 'intercept']
combos = combos[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])

# Vorhersagen machen
combos['admit_pred'] = result.predict(combos[train_cols])

print combos.head()

def isolate_and_plot(variable):
    grouped = pd.pivot_table(combos, values=['admit_pred'], index=[variable, 'prestige'],
                            aggfunc=np.mean)
    colors = 'rbgyrbgy'
    for col in combos.prestige.unique():
        plt_data = grouped.ix[grouped.index.get_level_values(1)==col]
        pl.plot(plt_data.index.get_level_values(0), plt_data['admit_pred'],
                color=colors[int(col)])

    pl.xlabel(variable)
    pl.ylabel("P(admit=1)")
    pl.legend(['1', '2', '3', '4'], loc='upper left', title='Prestige')
    pl.title("Prob(admit=1) isolating " + variable + " and presitge")
    pl.show()

isolate_and_plot('gre')
isolate_and_plot('gpa')
