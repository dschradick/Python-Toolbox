########## ANOVA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


#### Daten generieren
np.random.seed(0)
a = stats.norm.rvs(loc=50, scale=15, size=1000)
b = stats.norm.rvs(loc=50, scale=15, size=1000)
c = stats.norm.rvs(loc=50, scale=15, size=1000)
d = stats.norm.rvs(loc=50, scale=15, size=1000)
# ==> zuerst alle Gruppen mit selber Verteilung


#### ANOVA durchführen
result = stats.f_oneway(a, b, c, d)
result.statistic
result.pvalue
# => H0 kann nicht abgelehnt werden

## Daten ändern: Mittelwert von d von 50 => 51 ändern
d = stats.norm.rvs(loc=51, scale=15, size=1000)
result = stats.f_oneway(a, b, c, d)
result.pvalue
# => H0 kann abgelehnt werden



#### Post-Hoc Test: Tukey
from statsmodels.stats.multicomp import pairwise_tukeyhsd

x = np.array(['a', 'b', 'c','d'])
labels = np.repeat(x, [1000, 1000, 1000, 1000], axis=0)
values = np.hstack([a, b, c, d])
tukey = pairwise_tukeyhsd(endog=values,     # Daten
                          groups=labels,    # Gruppen
                          alpha=0.05)       # Signifikanz-Level

tukey.plot_simultaneous()    # Konfidenzintervalle der Gruppen
plt.vlines(x=49.57,ymin=-0.5,ymax=4.5, color="red")

tukey.summary()
