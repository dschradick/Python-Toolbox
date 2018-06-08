########## CLUSTERING - K-MEANS
# vs hierachisch => Anzahl der Cluster vorgegeben
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
samples = iris.data


##### SKALIEREN NICHT VERGESSEN!!!
scaler = StandardScaler()
scaler.fit(samples)
samples_scaled = scaler.transform(samples)


#### Besser: In Pipepline
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)
pl = make_pipeline(scaler,kmeans)
_ = pl.fit(samples)
labels = pl.predict(samples)
df = pd.DataFrame({'Cluster':labels,'Varieties':iris.target})
pd.crosstab(df.Cluster,df.Varieties)


#### Clustering normal durchführen
samples = samples[:,[0,2]]
model = KMeans(n_clusters=3)  # weil drei iris Species
_ = model.fit(samples)
# k-means speichert die centroids


#### Vorhersage
# => neue Samples können nun zu existierenden Clustern zugeordnet werden,
#    durch Bestimmung welcher centroid am nächsten
labels = model.predict(samples)
print(labels)


#### Visualisieren
xs = samples[:,0] # Sepal-length in Column 0
ys = samples[:,1] # Petal-length in Colum 2
centroids = model.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
plt.scatter(xs, ys, c=labels, alpha=0.5)
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()



#### Evaluierung des Clusterings

### Mit vorgegeben Klassen
# Cross-Tabulation
# => zeigt welche Samples in welchem Cluster sind
species = iris.target_names[iris.target]
df = pd.DataFrame({'Cluster':labels,'Species':species})
pd.crosstab(df.Cluster,df.Species) # => Cluster 2 = nur setosa; Virginica von 2 verschiedenen Clustern erfasst

### Ohne vorgegebene Klassen
# Metrik: Inertia / within-cluster sum-of-squares:
#   Misst wie weit nach außen verteilt sind die Cluster (niedirger ist besser)
#   => Distanz von jedem Sample zu centroid des Clusters (Minimierung ist Ziel von k-means)
# Ziel: Jedes clustering hat einen engen cluster
#       => beobachtungen sind eng zusammen und nicht weit voneinander verteilt

# Für jedes k plotten!
print(model.inertia_)

## Ellbogen-Kurve: Wahl des k
# Gutes Clustering hat enge clusters (niedriges inertia), aber nicht zuviele cluster
# "Ellbogen" im k-plot wählen
# =>  Punkt an dem inertia fängt langsamer abzunehmen
ks = range(1,10)
inertias = []
for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(samples)
    inertias.append(model.inertia_)

plt.plot(ks, inertias, '-o')
plt.xlabel('k - anzahl der cluster')
plt.ylabel('Inertia')
_ = plt.xticks(ks)
plt.show() # => 3 gute Wahl
