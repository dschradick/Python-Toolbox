########## FEATURE TRANSFORMATION
# Features haben häufig verschiedene Varianzen haben oder verschiedene Skalen benutzen
# Problem: einige Algorithmen (z.B. K-nearest neigbors) verwenden Distanz als Information für prediction
#   => größere Skalen haben ein ungerechtfertigt großen Einfluss auf das Modell
#      z.B: k-means: feature-varianz = feature-influence
# Lösung: Standardisierung
# => Subtraktion des arithmetischen Mittelwert + teilen durch die Standardabweichung
# In sklean:
# - StandardScaler
# - Normalizer
# - MaxAbsScaler
# - MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
import pandas as pd

#### Daten vorbereiten
wine = pd.read_csv('data/wine.csv')
target = wine.class_name
samples = wine.drop('class_name',axis=1)
samples = samples.drop('class_label',axis=1)

#### Einfaches Scaling
scaler = StandardScaler()
scaler.fit(samples)
samples_scaled = scaler.transform(samples)


#### K-Means & Pipeline
## Ohne scaling
model = KMeans(n_clusters=3)
_ = model.fit(samples)
labels = model.predict(samples)
df = pd.DataFrame({'Cluster':labels,'Varieties':target})
pd.crosstab(df.Cluster,df.Varieties)
 # => relativ schlechte Voraussage


## Mit Scaling
from sklearn.pipeline import make_pipeline
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)
pl = make_pipeline(scaler,kmeans)
_ = pl.fit(samples)
labels = pl.predict(samples)
df = pd.DataFrame({'Cluster':labels,'Varieties':target})
pd.crosstab(df.Cluster,df.Varieties)
# => bessere Voraussage
