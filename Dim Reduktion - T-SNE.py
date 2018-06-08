########## T-SNE
# = t-distributed stochastic neighbor embedding
# Mapped Samples auf 2d/3d Space
#  => repräsentiert dabei gut die Distanz zwischen den Samples
#
# Learning-Rate: rumprobieren - zwischen 50-200 wähen  (anderen wählen wenn alle aufeinander)
# => Werte auf den Achsen haben keine interpretierbare Aussage
#
# Insbesondere gut: wenige Samples mit vielen Features (z.B. Börse - wenige Kurse, viele Werte)
# Bsp: IRIS Datenset = 4 Dimensionen => nun nach 2 Dimensionen
from sklearn import datasets
iris = datasets.load_iris()
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


#### T-SNE anwenden
# learning_rate=50-200  (anderen wählen wenn alle aufeinander)
model = TSNE(learning_rate=100)
transformed = model.fit_transform(iris.data)
xs = transformed[:,0]
ys = transformed[:,1]

#### Visualisieren
plt.scatter(xs,ys,c=iris.target)


#### Wenn zuviele Labels
# => dann direkt im diagramm kennzeichnen (hier aber zuviele Datenpunkte)
for x, y, specie in zip(xs, ys, iris.target):
    plt.annotate(specie, (x, y), fontsize=10, alpha=0.75)
plt.show()
