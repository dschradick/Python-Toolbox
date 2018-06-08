########## PUNKTSCHÄTZUNG & KONFIDENZINTERVALLE
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math

#### Vorbereitungen
# Grundgesamtheit
population = stats.norm.rvs(loc=75, scale=20, size=1000, random_state=0)

#### Punktschätzung
# Punktschätzung ist Schätzung eines Parameters der Grungesamtheit
# auf Basis einer Stichprobe
sns.distplot(population);
population.mean()
# Sample erzeugen
np.random.seed(0)
sample = np.random.choice(a=population, size=100, random_state=0)
## Beispiel-Punktschätzung vornehmen: Mittelwert
print("Point Estimate:", np.mean(sample))  # Punktschätzung für den Mittelwert
population.mean() - sample.mean()          # Untschied zwischen echtem und geschätzem Wert


#### Konfidenzintervall
np.random.seed(0)
sample_size = 100
sample = np.random.choice(a = population, size = sample_size)
sample_mean = sample.mean()

## Mit stats
sample_stdev = sample.std()
sigma = sample_stdev / math.sqrt(sample_size)
confidence_interval = stats.t.interval(alpha = 0.95,
                 df= 99,
                 loc = sample_mean,
                 scale = sigma)
print("Confidence interval:", confidence_interval)

## Manuell - wahre SD nicht bekannt
t_critical = stats.t.ppf(q = 0.975, df=sample_size - 1); t_critical
sample_stdev = sample.std()
margin_of_error = t_critical * sigma
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
print("Confidence interval:", confidence_interval)


## Manuell - wahre SD bekannt
z_critical = stats.norm.ppf(q = 0.975) ; z_critical
pop_stdev = population.std()
margin_of_error = z_critical * (pop_stdev / math.sqrt(sample_size))
confidence_interval = (sample_mean - margin_of_error,
                       sample_mean + margin_of_error)
print("Confidence interval:", confidence_interval)




#### Weiteres

### Stichprobenverteilung des Mittelwerts
# = Sampling Distribution of the mean
np.random.seed(0)
point_estimates = []
for x in range(1000):
    s = np.random.choice(a=population, size=100)
    point_estimates.append(s.mean())
## Sampling Distribution anzeigen
sns.distplot(point_estimates);
# Abweichung des Mittelwerts der Sampling Distribution von wahren Mittelwert
population.mean() - np.array(point_estimates).mean()
## Standard Fehler
# SE einer Statistik ist die Standarabweichung für Schätzung von unbekanntem Parameter
# = Standardabweichung der Sampling Distribution
# 1. Als Abweichung der Sampling Distribution
np.std(point_estimates)
# 2. Schätzung
sample = np.random.choice(a=population, size=100)
sample.std() / math.sqrt(len(sample))


### Konfidenz-Intervalle visualisieren
np.random.seed(0)
sample_size = 100; intervals = [] ; sample_means = []
for sample in range(25):
    sample = np.random.choice(a= population, size = sample_size)
    sample_means.append(sample.mean())

    z_critical = stats.norm.ppf(q = 0.975)
    pop_stdev = population.std()
    stats.norm.ppf(q = 0.025)
    margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size))
    confidence_interval = (sample_mean - margin_of_error,
                           sample_mean + margin_of_error)
    intervals.append(confidence_interval)

plt.figure(figsize=(9,9))
plt.hlines(xmin=0, xmax=25, y=43.0023, linewidth=2.0, color="red")
plt.errorbar(x=np.arange(0.1, 25, 1), y=sample_means, yerr=[(top-bot)/2 for top,bot in intervals], fmt='o')
