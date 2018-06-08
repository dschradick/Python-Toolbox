########## PRINCIPAL COMPONENT ANALYSIS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

#### Daten einlesen
iris = datasets.load_iris()
wine = pd.read_csv('~/Documents/Data/wine.csv')
samples = wine[['total_phenols','od280']]


#### PCA durchführen
model = PCA(n_components=0.95) # oder n_components = 2 
## !!! n_components kann arg als float 0.0 and 1.0 prozent der varianz angegeben werden die erhalten bleiben soll !!!
_ = model.fit(samples)
transformed = model.transform(samples)

# Normalerweise vorher: Skalierung
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler,pca)
_ = pipeline.fit(samples)

### PCs
print("Principal Components: \n", model.components_)
samples_correlation, pvalue = pearsonr(samples['total_phenols'],samples['od280'])
pca_correlation, pvalue = pearsonr(transformed[0],transformed[1])
print("Samples Pearson: {0:f}".format(samples_correlation))
print("PCA Pearson: {0:f}".format(pca_correlation))


#### Visualisieren
transformed = pd.DataFrame(transformed)
samples = pd.concat([samples,wine[['class_name']]],axis=1)
transformed_df = pd.concat([transformed,wine[['class_name']]],axis=1)
transformed_df.columns = samples.columns

## Original-Daten + erste 2 PCs als Pfeil
_ = sns.lmplot(data=samples,x='total_phenols',y='od280',hue='class_name',fit_reg=False)
mean = model.mean_
# =>  Der Punkt an dem später neu ausgerichtet wird
first_pc = model.components_[0,:]
second_pc = model.components_[1,:]
_ = plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.05)
_ = plt.arrow(mean[0], mean[1], second_pc[0], second_pc[1], color='red', width=0.05)
plt.show()

## Transformierte Daten
_ = sns.lmplot(data=transformed_df,x='total_phenols',y='od280',hue='class_name',fit_reg=False)
plt.show()
# =>  Rotiert und geshiftet (siehe wo 0-Punkt)


#### Intrinsische Dimension bestimmen
##  Problem: > 2 Dimensionen und man kann intrinische Dimension nicht durch Scatterplot bestimmmen
# Lösung: die erklärte Varianz durch die PCA-Features betrachten => wieviele sind signifikant?
features = range(model.n_components_)
plt.bar(features,model.explained_variance_)
plt.xlabel('PCA Feature'); plt.ylabel('Variance');
_ = plt.xticks(features)
plt.show()
#!! => nur 2 Features mit signifkanter (> 0.5 (benötigt vorige Skalierung?)) Varianz => intrinsische Dimension = 2




#### Weiteres: Intrinsische Dimension grafisch bestimmen
from mpl_toolkits.mplot3d import Axes3D
samples = iris.data[:,[0,1,3]]          # nur drei features
samples = samples[iris.target == 1]     # nur versiolor betrachten
# => 3D Plot

### Visualisierung
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.margins(0.4)
_ = ax.set_xlabel('Sepal Length'); _ = ax.set_ylabel('Sepal Length'); _ =ax.set_zlabel('Petal Length')
_ = ax.scatter(samples[:,[0]],samples[:,[1]],samples[:,[2]])
# => alle ungefährt auf einer flachen 2-dimensionalen Ebene
#    kann durch 2 Features approximiert werden (ohne viel Information zu verlieren)
#    => Intrinsische Dimension = 2
# Nun PCA die Samples rotieren und Shiften!

model = PCA()
samples = model.fit_transform(samples)
mean = model.mean_
first_pc = model.components_[0,:] ; second_pc = model.components_[1,:] ; third_pc = model.components_[2,:]

_ = ax.quiver3D(mean[0], mean[1], mean[2], first_pc[0], first_pc[1],first_pc[2], color='red') #, length=model.explained_variance_[0]*6)
_ = ax.quiver3D(mean[0], mean[1], mean[2], second_pc[0], second_pc[1],second_pc[2], color='red') # , length=model.explained_variance_[1]*6)
_= ax.quiver3D(mean[0], mean[1], mean[2], third_pc[0], third_pc[1],third_pc[2], color='red') # , length=model.explained_variance_[2]*6)
plt.show()
# PCA Features plotten
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('PCA Feature 0'); ax.set_ylabel('PCA Feature 1'); ax.set_zlabel('PCA Feature 2')
_ = ax.scatter(samples[:,[0]],samples[:,[1]],samples[:,[2]])
plt.show()
