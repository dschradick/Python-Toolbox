########## BOOTSTRAP - EINSTICHPROBEN TEST
# Bootstrap testing analog zu 1-sample t-test
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import Bootstrap_Helfer as bh

#### Hypothesen
# mu_0 = 50
# H0: mu <= mu_0
# HA: mu > mu_0
# Alpha = 0.05

#### Daten erzeugen
mu_0 = 50
sample = np.array([51,69,49,61,55,55])
np.mean(sample)  # 56.66
np.std(sample)   #  6.67

#### Erzeugen der Daten unter der Null-Hypothese:
# => value ist der Mittelwert
h0_data = sample - np.mean(sample) + mu_0


#### Teststatistik = Mittelwert von gegebenem Sample - value
def diff_from_value(data,value):
    return np.mean(data) - value


#### Berechnen der test-statistik der beobachteten Daten
observed_difference = diff_from_value(sample,mu_0);
observed_difference


#### Simulieren der Sampling Distribution unter H0
diff_from_mu0 = lambda x: diff_from_value(x,mu_0)
bs_replicates = bh.draw_bs_reps(h0_data, diff_from_mu0, 10000)

## A. CRITICAL REGION APPROACH

#### Visualisieren
# Sampling Distribution, kritische Region
# und Test-statistik der beoachteten Daten
critical = np.percentile(bs_replicates,95)
_ = plt.hist(bs_replicates, bins=25);
_ = plt.axvline(critical, color='r', linestyle='dashed', label='Critical Value') ;
_ = plt.axvline(observed_difference, color='k', linestyle='solid', label='Observed Test-Statistic')
_ = plt.legend()
# => Wert liegt im kritischen Bereich
# => H0 kann abgelehnt werden 


## B. P-VALUE APPROACH

#### p-Wert berchnen
# Anteil der Replicas, welche größer sind als die beobachtete Teststatistik
p_value = np.sum(bs_replicates >= observed_difference) / 10000
p_value  #  0.0121
# => 0.0121 < 0.05
# => kann auschliessen, dass der echte Mittelwert kleiner als value

## C. CI-APPROACH
# => hier NICHT anwendedbar, weil hypothese einseitig
# - aber angenommen H_a wäre: mu <> mu_0

# 95% Konfidenz-Intervall
bs_replicates = bh.draw_bs_reps(sample,np.mean,10000)
conf_int = np.percentile(bs_replicates,[2.5,97.5])
print(conf_int)  # [52.         62.33333333]
# => Mu_0 ist nicht im 95%CI => H_0 ablehnen