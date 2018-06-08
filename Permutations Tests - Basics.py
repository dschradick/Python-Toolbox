########## PERMUTATIONS TESTS - BASICS
# Funktionsweise:
# Hypothese: analog z.B. zum 2-sample t-test
# Bsp: Verhältnis demokratische / republikanisch Wähler
#      in Pennsylvania und Ohio ist gleich
#
# Idee: Simulieren, wie die Daten aussehen würden,
#       wenn die Hypothese zutrifft und mit Beobachteten vergleichen
#
# Umsetzung:
# 1. Daten der beiden Vektoren (Länge n,m) zusammenfügen
# 2. Permutation: Daten permutieren
# 3. Relabeling:
#     - ersten n Elemente als erste Größe (Pennsylvania) markieren
#     - m folgenden als zweite Größe (Ohio) labeln
#   => Messung wiederholen auf generierten Daten - unter Annahme H0
# 4. Sampling Distribution für Messungen unter H0 erstellen und
#    vergleichen mit beobachteter, tatsächlicher Messung
#    => wie wahrscheinlich ist beobachtete Messung unter H0
#    => Histogramm und quantifizierung durch P-Wert
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import Bootstrap_Helfer as bh


#### Daten einlesen und vorbereiten
election_all = pd.read_csv('~/Documents/Data/2008_all_states.csv',header=0)
election_all.groupby('state')['dem_share'].mean().sort_values()
# => zwei recht ähnliche - PA, OH - und einer - MD - der etwas(!) mehr hat
dem_share_PA = election_all[election_all.state == 'PA']['dem_share'] # Pennsylvania
dem_share_OH = election_all[election_all.state == 'OH']['dem_share'] # Ohio
dem_share_MD = election_all[election_all.state == 'MD']['dem_share'] # Maryland


#### Daten betrachten
dem_share_PA.mean() # 45.476417910447765
dem_share_OH.mean() # 44.31818181818181
dem_share_MD.mean() # 50.038333333333334


x_1, y_1 = bh.ecdf(dem_share_PA)
x_2, y_2 = bh.ecdf(dem_share_OH)
x_3, y_3 = bh.ecdf(dem_share_MD)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='yellow')
_ = plt.plot(x_3, y_3, marker='.', linestyle='none', color='blue')
plt.show()
# => Pennsylvania und Ohio ziemlich gleich,
#    Maryland höherer Anteil Demokraten - signifikant?!


##### Zusammenfügen, Permutieren & Relabeln
### Bsp: Pennsylvania und Ohio
#
## 1. Zusammenfügen
dem_share_both = pd.concat([dem_share_PA,dem_share_OH])
## 2. Permutieren
dem_share_perm = np.random.permutation(dem_share_both)
# 3. Relabeling => Daten unter H0
perm_sample_PA = dem_share_perm[len(dem_share_PA)]
perm_sample_OH = dem_share_perm[:len(dem_share_PA)]

##### Vergleich von Permutations-Replicate mit Statistik der beobachten Daten
# Permutations-Replicate = Statistik der Permuation (Daten unter H0)
# Test-Statistik: Differenz des Mittelwerts
#    = Basis zum Vergleich was H0 voraussagt und dem was tatsächlich beobachtet wurde
#      => zum Vergleich zwischen beobachteten & simulierten Daten
np.mean(perm_sample_PA) - np.mean(perm_sample_OH)
# => 4,3 = Permutations-Replicatee
observed_difference = np.mean(dem_share_PA) - np.mean(dem_share_OH)
print(observed_difference)
# => 1,5 = Test-Statistik der beobachteten Daten


#### oberen Vorgang 10000 mal wiederholen
diff_of_means = lambda data_1, data_2: np.mean(data_1) - np.mean(data_2)
perm_replicates = bh.draw_perm_reps(dem_share_PA, dem_share_OH, diff_of_means, size=10000)

#### Visualisieren
critical = np.percentile(perm_replicates,95)
_ = plt.hist(perm_replicates, bins=20, normed=True);
_ = plt.axvline(observed_difference, color='k', linestyle='solid')
_ = plt.axvline(critical, color='r', linestyle='dashed') ;
# => nicht im kritischen Bereich
# => H0 kann nicht abgelehnt wrden




#### Im Fall von echtem Unterschied
# Bsp: Maryland und Ohio
# H0: Es besteht kein Unterschied im Wahlverhalten
# HA: Maryland hat höheren Anteil an demoratischen Wählern
perm_replicates = bh.draw_perm_reps(dem_share_OH, dem_share_MD, diff_of_means, size=10000)
observed_difference = np.mean(dem_share_MD) - np.mean(dem_share_OH)
print(observed_difference)


#### Visualisieren
critical = np.percentile(perm_replicates,95)
_ = plt.hist(perm_replicates, bins=20, normed=True);
_ = plt.axvline(observed_difference, color='k', linestyle='solid')
_ = plt.axvline(critical, color='r', linestyle='dashed') ;
# => Beobachte Differenz im kritischen Bereich
#    Berechnung des P-Wertes wäre an sich nicht notwendig


#### P-Wert berechnen
p_value = np.sum(perm_replicates >= observed_difference) / 10000
p_value # p = 0.0127  < alpha = 0.05
# 0.0127  < alpha = 0.05
# => 1% der der Werte sind extremer als die Teststatistik der beobachten Differenz
# => H0 kann agelehnt werden, in Maryland wird mit größerem Anteil als Ohio demokraten gewählt
#    => Unterschied läßt sich nicht durch Zufall erklären (bei alpha=0.05)
