########## CLUSTERING - HIERACHISCH
# Vs. k-Means - Anzahl der Cluster muss nicht vorher festgelegt werden
# Repräsentation & Darstellung durch Dendogramm
# Funktionsweise:
#  1. Jedes Sample ist sein eigener Cluster
#  2. In jedem Schritt werden die zwei gemergt, welche am nächsten zusammen sind
#  3. Solange bis alle Samples ein Cluster
#  => "Agglomeratives" Clustering
#  => "Divisive" Clustering" funktioniert genau anders herum
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#### Daten vorbereiten
samples = pd.read_csv('data/eurovision.csv')
names_pd = pd.read_csv('data/eurovision_country.csv')
names = names_pd['0'].values

#### Hierarchisches clustering durchführen
# complete => Distanz zwischen Clusters ist die max. Distanz zwischen ihren Samples (weitest mögliche Distanz zwischen Punkten zwischen Clusters)
# single   => Distanz zwischen Clusters ist die min. Distanz zwischen ihren Samples
### Complete Linkage
mergings = linkage(samples, method='complete')
_ = dendrogram(mergings, labels=names, leaf_rotation=90, leaf_font_size=6)
plt.show()
### Single Linkage
mergings = linkage(samples, method='single') # => distanz = Distanz zwischen den nähesten Punkten
_ = dendrogram(mergings,labels=names,leaf_rotation=90,leaf_font_size=6)
plt.show()



#### Cluster erzeugen
# Höhe im Dendrogramm = Distanz zwischen den gemergten Clustern
# Cluster bilden durch horizontalen Cut
#  => Crosstablution auf verschiedenen Stufen betrachten
# Extrahieren von Cluster Labels & Crosstabbing

## Bilden von Clustern durch Cut auf Höhe 15
labels = fcluster(mergings, 15, criterion='distance')

## Clustering ausgeben
pairs = pd.DataFrame({'labels':labels,'countries':names})
print(pairs.sort_values('labels').head())

## Crosstabulation anzeigen
pd.crosstab(pairs['labels'],pairs['countries'])
