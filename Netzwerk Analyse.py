########## NETZWERK ANALYSE
import networkx as nx
import matplotlib.pyplot as plt

#### Graph erstellen
G = nx.Graph()
D = nx.DiGraph()        # Directed Graph
M = nx.MultiDiGraph()   # Mehrere Kanten


#### Nodes hinzufügen
G.add_nodes_from([1,2,3])
G.nodes()


#### Kanten hinzufügen
G.add_edge(1,2)
G.add_edge(1,3)
#G.edge[1][2]['weight'] = 2  # Kanten-Beschriftung
G.edges()


#### Metadaten zu Node hinzufügen
G.node[1]['label'] = 'blue'
G.nodes(data=True)


#### Interessante Nodes bekommen !!!
noi = [n for n, d in T.nodes(data=True) if d['occupation'] == 'scientist']
## Interessante Kanten
eoi = [(u, v) for u, v, d in T.edges(data=True) if d['date'] < date(2010, 1, 1)]

## Über Kanten iterieren
for u, v, d in T.edges(data=True):
    if 293 in [u, v]: # also node 293 entweder u,v
        # Set the weight to 1.1
        T.edge[u][v]['weight'] = 1.1

## Über Knoten iterieren
degrees = [len(T.neighbors(n)) for n in T.nodes()]
print(degrees)

## Subgraph erstellen
sub_graph = G.subgraph(list_of_nodes)

## Maximale Cliquen finden => jeder mit jedem verbunden
nx.find_cliques(G)
largest_clique = sorted(nx.find_cliques(G), key=lambda x:len(x))[-1] # größte maximale clique



#### Graph zeichnen
nx.draw(G)
plt.show()

## Matrix-Plot
import nxviz as nv
m = nv.MatrixPlot(G)
m.draw()
plt.show()

## Arc-Plot
ap = nv.ArcPlot(G)                                  # zusätzliche Args: node_color='category',node_order='category')
ap.draw()
plt.show()

## Cricos-Plot
from nxviz import CircosPlot
c = CircosPlot(G)
c.draw()
plt.show()


#### Zentralität
G.neighbors(1)                                       # liefert Nachbarn
nx.degree_centrality(G)                              # liefert centrality-score (keine self-loops)?
nx.betweenness_centrality(G)                         # liefert nicht higly-connected sondern BOTTLE-NECKS (WICHTIGE USER)
                                                     # zB. Leute zwischen Demokraten und Republikanern
max_dc = max(list(nx.degree_centrality(G).values())) # Maximaler Degree of centrality

## Größten Kollaboratoren
max_dc = max(nx.degree_centrality(G))
prolific_collaborators = [n for n, dc in nx.degree_centrality(G).items() if nx.degree(G,n) == max_dc]

## Degree-Historgram
plt.hist(list(nx.degree_centrality(G).values()))
plt.show()

## Dreiecks Recommendation
from itertools import combinations
def is_in_triangle(G, n):
    in_triangle = False
    for n1, n2 in combinations(G.neighbors(n),2):
        if G.has_edge(n1,n2):
            in_triangle = True
            break
    return in_triangle
