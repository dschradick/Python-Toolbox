########## CHI2-TEST
import numpy as np
import pandas as pd
import scipy.stats as stats


#### Verteilungs- / Anpassungstests / goodness-of-fit test
## Daten generieren
national = pd.DataFrame(["white"]*100000 + ["black"]*50000 + ["asian"]*15000)
city = pd.DataFrame(["white"]*900 + ["black"]*450 +["asian"]*130)

## Counts
national_table = pd.crosstab(index=national[0], columns="count")
city_table = pd.crosstab(index=city[0], columns="count")

## Chi^2 statistik berechnen
observed = city_table
national_ratios = national_table / len(national)   # Verhältnisse der Grundgesamtheit
expected = national_ratios * len(city)             # Erwartete Counts
expected
observed

## Manuell
chi_squared_stat = (((observed-expected)**2)/expected).sum() 
chi_squared_stat
crit = stats.chi2.ppf(q = 0.95, df = 2); crit # Freiheitsgrade = Anzahl der Kategorien - 1
p_value = 1 - stats.chi2.cdf(x=chi_squared_stat, df=4) ; p_value

## Mit stats
stats.chisquare(f_obs = observed, f_exp = expected)
# => es kann nicht widerlegt werden, dass es dasselbe Verhältnis wie Grundgesamtheit



#### Test auf Unabhängigkeit

## Testdaten erzeugen
np.random.seed(10)
voter_race = np.random.choice(a= ["asian","black","hispanic","other","white"],
                              p = [0.05, 0.15 ,0.25, 0.05, 0.5],
                              size=1000)
voter_party = np.random.choice(a= ["democrat","independent","republican"],
                              p = [0.4, 0.2, 0.4],
                              size=1000)
voters = pd.DataFrame({"race":voter_race,"party":voter_party})
voter_tab = pd.crosstab(voters.race, voters.party, margins = True)
voter_tab


### Observed und expected berechnen
# Unterschied zu normalen Anpassungstest:
#  => expected wird für 2D tabelle anstatt für 1D array berechnet
stats.chi2_contingency(observed= observed)
# => liefert gleichzeitig auch expected frequencies
#    basierend auf den Randverteilungen
