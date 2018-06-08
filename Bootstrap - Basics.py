########## BOOTSTRAP
# Simuliert wiederholte Datenmessung
# Ziel: welchen Wert kann man erwarten, wenn man immer und wieder - unendlich -
#       Messung wiederholt, um zu möglichst gute Folgerung zu machen
#
# Terminologie:
# Resampling:             Simuliert eine widerholte Messung, wenn man keine weiteren Daten hat / Messen kann
# Bootstrapping:          Verwendung von resampled data, um statische Inferenz vorzunehmen
# "Bootstrap Sample":     Jedes resampletes Array heisst Bootstrap Sample
# "Bootstrap Replicate":  Statistik, welche vom resamplten array gewonnen wurde
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


#### Sampling mit zurücklegen
sample = np.array([1,2,3,4,5])                           # Echte gemessene Daten
sample.mean()   # => 3.0                                 # Statistik auf echtem Sample
resample = np.array([4,3,5,1,5])                         # "Bootstrap Sample" manuell
resample = np.random.choice(sample,size=len(sample))     # "Bootstrap Sample" automatisch generiert
resample.mean() # => 3.6                                 # "Bootstrap Replicate" => Statistik von "Bootstrap Sample"


import Bootstrap_Helfer as bh
bh.bootstrap_replicate_1d(sample,np.mean)               # die letzten beiden zusammen
bh.draw_bs_reps(sample,np.mean, 1000)                   # 1000 mal wiederholt => 1000 bootstrap relicates


#### Parameter-Schätzung
#### Beispiel: mehrere Messung der Lichtgeschwindigkeit
# 100 Messungen
speed_of_light = pd.read_csv('~/Documents/Data/michelson_speed_of_light.csv',header=0)['velocity of light in air (km/s)'].values
np.mean(speed_of_light)   # 299852.4
# => Schätzung des Mittelwerts basiert auf Sample
np.std(speed_of_light)    # 78.61450247886836
# => Standardabweichung
_ = plt.hist(speed_of_light, bins=30)


#### Simulation der Sampling Distribution der Grundgesamtheit
# => Standard-Fehler des Mittelwert der Grundgesamtheit entspricht dem der BS-Replicates
bs_replicates = bh.draw_bs_reps(speed_of_light,np.mean,1000)
_ = plt.hist(bs_replicates, bins=30)
bs_replicates.mean() # 299852.39195
                     # => Schätzung des Mittelwerts basierend auf Bootstrapping
speed_of_light.std() # => 78  Standardabweichung des Samples
# 1. Schätzung des Standard-Error auf Basis der  Sampling Distirbution
bs_replicates.std()  # => 7.68   
# 1. Schätzung des Standard-Error auf Basis des CLT
np.std(speed_of_light) / np.sqrt(len(speed_of_light))  # => 7.68


#### Bootstrap Konfidenz-Intervall für Parameter
conf_int = np.percentile(bs_replicates,[2.5,97.5])
print(conf_int) # [299836.5    299867.7025]


#### Lineare Regression
## Bootstrapping der Parameter der Regression
x = np.arange(0,100)
y = np.array([3 + value * 2 + round(np.random.rand() * 20) for value in x])
plt.scatter(x,y)
slope_reps, intercept_reps = bh.draw_bs_pairs_linreg(x,y,1000)
## Konfidenzintervalle
intercept_conf_int = np.percentile(intercept_reps,[2.5,97.5])
slope_conf_int = np.percentile(slope_reps,[2.5,97.5])
print(intercept_conf_int) # [11.52768478 16.05255611]
print(slope_conf_int)     # [1.94681907 2.02361893]
