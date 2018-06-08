########## STATISTISCHE TESTS - T-TEST
import numpy as np
import pandas as pd
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
import seaborn as sns


### Vorbereitungen
np.random.seed(0)


### Zufallszahlen generieren
sample_size = 1000
x = stats.norm.rvs(loc=50, scale=5, size=sample_size, random_state=0)
sns.boxplot(x);
np.mean(x)
np.std(x)
np.median(x)


### One-Sample t-test  
result = stats.ttest_1samp(x, 51) 
result.statistic # -7.853656368036508
result.pvalue    # 1.0389280338119273e-14
# ==> p < 0.05 ==> können H0 ablehnen => echte Mittelwert der Grundgesamtheit verschieden von 51

## Kritische Werte
stats.t.ppf(q=0.025, df=sample_size-1) # -1.962341461133449
stats.t.ppf(q=0.975, df=sample_size-1) # 1.9623414611334487

## Wahrscheinlickeit einen mindestens so extremen Wert zu sehen
print('%f' % (stats.t.cdf(x=result.statistic, df=sample_size-1) * 2))
#  => ~0%

## 95% Konfidenz-Interval
stats.t.interval(0.95, df = len(x)-1, loc = np.mean(x), scale= stats.sem(x))


#### Two-sample t-test
# equal_var=False => Welch t-test
# => verlässlicher bei verschiedener varianz und verschiedene sample size - nur normalität gefordert
#    allgemein wenn unpaired und nicht überlappende samples
# (student's t-test nimmt zusätzlich gleiche varianz an)
y = stats.norm.rvs(loc=51, scale=5, size=sample_size, random_state=0)
stats.ttest_ind(a = x, b = y, equal_var=False)
# => H0 kann abgelehnt werden (bei alpha=0.5)


#### Paired t-test
x_after = x + stats.norm.rvs(loc=.6, scale=5, size=sample_size, random_state=0)
stats.ttest_rel(a = x, b = x_after)
# => H0 kann abgelehnt werden (bei alpha=0.5)
