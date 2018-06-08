########## NMF - NON-NEGATIVE-MATRIX-FACTORIZATION
# Technik zur Dimensionality Reduction
#  => NMF Modelle sind im Gegensatz zu PCA interpretierbar
#  => einfach zu verstehen und zu erklären
# Bedingung: Feature dürfen nicht negativ sein (>= 0)
#  => z.B. Kauf-Historie im Shop, Wort-Frequenz in Dokument, Image-Codierung, Audio-Spectograms
# Repräsentation: Dokumente werden als Kombinationen von Topics/Themes(jeweils Menge von Wörtern) ausgedrüclickthrough_A
# Bsp: Sprache: "Basic ist eine tolle Programmiersprache zum Lernen" = 0.6 * Progammiersprachen('Basic','C',..) + 0.4 * Bildung('lernen','Schule',..) + ...
#      Bilder: als combination von Pattern  "_|"" = 0.9 * "_" + 0.8 * "|"
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF


#### Daten einlesen
df = pd.read_csv('data/wikipedia/wikipedia-vectors.csv', index_col=0) # Spalte und Reihen vertauscht
df_words = pd.read_csv('data/wikipedia/wikipedia-vocabulary-utf8.txt',header=None)
articles = csr_matrix(df.transpose()) # transpose, wegen csv aufbau sonst -  sonst 13.000 Spalte (=Wörter)
titles = list(df.columns)

#### NMF anwenden
# Verwendet numpy-array und csr-matrix
model = NMF(n_components=6)
_ = model.fit(articles)
nmf_features = model.transform(articles) # Die Principal Components
#labels = pipeline.predict(articles)
#df2 = pd.DataFrame({'label': labels, 'article': titles})
#print(df2.sort_values('label')) # => cluster entsprechen den themen


### NMF Komponenten
# Anzahl muss immer explizit angegeben werden
# NMF hat wie PCA auch Principal Components, wobei deren Dimension = Dimension der Samples
# => Einträge sind PC und immer nicht-negativ
print(articles.shape)
print(model.components_.shape)

## NMF Features
# => NMF-Feature Werte sind ebenfalls nicht-negativ
print(nmf_features.shape) # Reduzierten Features = 2


##!!! Interpretation
# NMF lernt die interpretierbaren Teile in den Daten
# NMF Komponenten repräsentieren die (gelernten) Topics
# Bei Bildern: Jede Zeile ein Bild mit sovielen Dimensionen wie Anzahl der Pixel
#  => NMF lernt die Patterns in den Bildern
df = pd.DataFrame(nmf_features,index=titles)
print(df.loc['Anne Hathaway'])
print(df.loc['Denzel Washington'])
#! => beide haben hohen NMF Feature 3 Wert (also Topic 3 - z.B. Acting)
#     -> beide können hauptsächlich mit der dritten NMF Komponente rekonstruiert werden
# Wörter in 3. Komponente Anzeigen
components_df = pd.DataFrame(model.components_,columns=df_words[0])
print(components_df.shape)
component = components_df.iloc[3] # 3, weil Denzel und Anna beide durch NMF Komp 3 repräsentiert sind
print(component.nlargest()) # => Interpretation

## Sample Rekonstruktion
# NMF-Features Können verwendet werden die Samples zu rekonstruieren (Kombination der Feature-Werten mit Komponenten)
# Multipliziere Komponenten mit Feature-Values und ergebnisse aufaddieren
#  entspricht Matrix Multiplikation => "Matrix Factorization" in NMF
# rec = np.dot( nmf_features[2],model.components_)? (problem ist sparsematrix)
from sklearn.decomposition import NMF
R = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]
R = np.array(R)
nmf = NMF(n_components=2)
W = nmf.fit_transform(R);
H = nmf.components_;
nR = np.dot(W,H)

for i,element in enumerate(nR):
    for j,element2 in enumerate(element):
        nR[i][j] = round(nR[i][j])
print (nR)


### NMF Recommender System (mit Cosine Similarity)
# Zwei Dokument können diesselbe Aussage haben, aber durch mehr sinnlose Wöter schwächer sein
#  => Feature Werte können dann verschieden sein, aber die verschiedenen Versionen liegen auf derselbe Gerade die durch den Ursprung verläuft
# Demensprechend den Winkel zwischen den Feature-Vektoren vergleichen mittels "Cosine-Similarity"
# => hoher Wert = hohe Ähnlichkeit (von 0 bis 1)
from sklearn.preprocessing import normalize

model = NMF(n_components=6)
nmf_features = model.fit_transform(articles) # Die Principal Components
norm_features = normalize(nmf_features)

# Cosine Similarities
current_article = norm_features[23,:]
similarities = norm_features.dot(current_article)
print(similarities) # => cosine similarities

# Similarities mit Titeln durch DataFrame labeln
df = pd.DataFrame(norm_features,index=titles)
#current_article = df.loc['Dog bites man']
similarities = df.dot(current_article)
print(similarities.nlargest())  # => findet die Artikel mit der höchsten Similarity
