########## WORD FREQUENCY ARRAYS
# Alternative Implementierung von PCA
# Jede Reihe entspricht einem Dokument, jede Spalte einem Wort aus einem fixen Vokabular
#    => Inhalt der Zelle = Wort Frequenz durch tf-idf
# tf = Term Frequency = Frequenz des Worts in Dokument
# idf = Inverse Document Frequency => reduziert den Einfluss von häufigen Wörter wie "the"
# Es kommen längst nicht alle Wörter in jedem Dokument vorher
#    => häufigsten der Wert 0 vertreten => sparse array
#    => CSR-Matrix: Sparse Matrx - speichert nur die nicht-null Entries -> spart Platz
# Aber: csr_matrix nicht von sklearn PCA unterstützt -> TruncatedSVD
#
# WORKFLOW:
# Documents -> TfidfVectorizer (erzeugt csr_matrix) -> PCA mittels TruncatedSVD -> Algorithmus (z.B. KMeans)
#
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

# Sparse-Matrix (csr_matrix) erstellen mittels TfidfVectorizer
documents = np.array(['cats say meow', 'dogs say woof', 'dogs chase cats'])
tfidf = TfidfVectorizer()                  # transformiert Liste von Dokumenten in eine csr_matrix
csr_mat = tfidf.fit_transform(documents)   # csr_matrix anzeigen
print(csr_mat.toarray())
print(tfidf.get_feature_names())           # Spalte des Arrays entsprechen Wörtern

# PCA mittels TruncatedSVD auf die csr_matrix anwenden
model = TruncatedSVD(n_components=3)
model.fit(csr_mat)                         # documents ist csr_matrix
transformed = model.transform(csr_mat)

# In Pipline anwenden: (TfidfVectorizer benutzen)
df = pd.read_csv('data/wikipedia/wikipedia-vectors.csv', index_col=0) # Spalte und Reihen vertauscht
df_words = pd.read_csv('data/wikipedia/wikipedia-vocabulary-utf8.txt',header=None)

articles = csr_matrix(df.transpose()) # transpose, da csv so aufgebaut -  sonst 13.000 Spalte (=Wörter)
titles = list(df.columns)

svd  = TruncatedSVD(n_components=50)
kmeans = KMeans(n_clusters=6)
pipeline = make_pipeline(svd,kmeans)

_ = pipeline.fit(articles)
labels = pipeline.predict(articles)
df2 = pd.DataFrame({'label': labels, 'article': titles})
print(df2.sort_values('label')) # => cluster entsprechen den themen
