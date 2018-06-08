########## MATRIX FACTORIZATION
# Gute Zusammenfassung Matrix Algebra:
# https://explained.ai/matrix-calculus/index.html

#### MATRIX FACTORIZATION
## Analog zur normalen Zerlegung von Zahlen in Faktoren
# z.B. C = A * B  =>  10 = 5 * 2
import numpy as np
from numpy import array, diag
from numpy.linalg import norm, inv

##### BASICS
#### EINFACHE MULTIPLIKATION
A = array([
  [1, 2],
  [3, 4],
  [5, 6]])

B = array([
    [1, 2],
    [3, 4]])

## Dimensionen
A.shape
B.shape
## Multiplikation
C = A.dot(B)
C.shape
print(C)



### QR Zerlegung
# A = Q*R
# - A = m x n Matrix
# - Q = m x m orthonormal Matrix
# - R = m x n upper triangle Matrix
# 3 x 2
from numpy.linalg import qr
A = array([
  [5, 2],
  [3, 4],
  [5, 6]])

Q, R = qr(A,'complete')
A.shape
Q.shape
R.shape
norm(Q[:,0])        # Länge 1
Q[:,0].dot(Q[:,1])  # Produkt = 0 => orthogonal
print(Q)
print(R)
# Rekonstruktion
B = Q.dot(R)
print(B)

### Cholesky Zerlegung
# A = L·L^T  oder =  U^T·U
# A = symmetrische (postive definite) Matrix
# L = lower triangle Matrix
# U = upper triangle Matrix
# => verwendet zum lösen von linear least square
from numpy.linalg import cholesky
### Symmetrische Matrix
A = array([
  [2, 1, 1],
  [1, 2, 1],
  [1, 1, 2]])
print(A)
# factorize
L = cholesky(A)
U = L.T
print(L)
print(L.T)
# reconstruct
B = L.dot(L.T
C = U.T.dot(U)
print(B)
print(C)



#### Eigenwertzerlegung
# zerlegt eine quadratische Matrix in eine Menge von Eigenvektoren und Eigenwerten
# hilft Eingeschaften der Matrix zu verstehen - analog zu Primzahl-Faktorisierung von Zahlen
#
### Eigenvektor
# v ist Eigenvektor einer Matrix A, wenn (Eigenwert-Gleichung):
#
#   A * v = λ * v
#
# wobei
#   A = quadratische Matrix, welche zerlegt werden soll
#   v = Eigenvektor der Matrix
#   λ = Eigenwert (skalar)
#
# - Matrix
#    + besitzt soviele Eigenvektoren wie Dimensionen
#    + ist positiv/negativ definit, wenn alle Eigenwerte positiv/negativ
# - sind Einheitsvektoren (Länge = 1) & Spaltenvektoren
#
# - Wenn v ein Eigenvektor:
#    + dann hat v dieselbe Richtung wie der Vektor A*v
#    + Vektor A*v ist dann λ mal das original v
#    + λ besagt, ob v durch Multipliation
#      verlängert, verkürzt oder Richtung geändert hat

from numpy.linalg import eig
A = array([
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]])
values, vectors = eig(A)
first_eigenvector = vectors[:,0]
first_eigenvalue = values[0]

## Überprüfen ob das stimmt
# A * v
print(A.dot(first_eigenvector))              # [ -3.73863537  -8.46653421 -13.19443305]
# λ * v
print(first_eigenvalue * first_eigenvector)  # [ -3.73863537  -8.46653421 -13.19443305]


### Zerlegung
# Eine quadratische Matrix kann als Produkt von Eigenvektoren und Eigenwerten dargestellt werden
#
#   A = Q · Λ · Q^T
#
# wobei
#   Q = Matrix bestehend aus Eigenvektoren
#   Λ = (gross Lambda) diagonale Matrix bestehend aus Eigenwerten
eig_values, eig_vectors = eig(A)
print(eig_vectors)
# [[-0.23197069 -0.78583024  0.40824829]
 # [-0.52532209 -0.08675134 -0.81649658]
 # [-0.8186735   0.61232756  0.40824829]]
eig_values
# array([ 1.61168440e+01, -1.11684397e+00, -1.30367773e-15])


### Rekonstruktion der ursprünglichen Matrix A
Q = eig_vectors
#Lambda = diag(eigen_values)
Lambda= array([
  [eig_values[0], 0,             0            ],
  [0,             eig_values[1], 0            ],
  [0,             0,             eig_values[2]]])
R = inv(Q)  # warum nicht transpose?

B = Q.dot(Lambda).dot(R)
print(B)


## PCA mit Eigenwertzerlegung
from numpy import mean, cov
from numpy.linalg import eig

A = array([
  [1, 2],
  [3, 4],
  [5, 6]])

## Zentrieren der Daten
# => Mittelwerte der Spalten berechnen und abzierehn
M = mean(A, axis=0)
C = A-M
## Kovarianz-Matrix berechnen
V = cov(C.T)  # Warum transponiert??

## Faktorisieren der Matrix
values, vectors = eig(V)
## Eigenvektoren und Eigenwerte
print(vectors)
print(values)
## Daten auf reduzierten Raum projezieren
P = vectors.T.dot(C.T)
print(P.T)






#### Singulärwertzerlegung 
# SVD liefert ähnliche Informationen wie Eigenwertzerlegung, ist aber genereller anwendbar
# => funktioniert für alle rechteckigen Matrizen (vs quadratisch)
#
# Benutzt für:
#  - Berechnung der (pseudo-) inversen Matrix (z.B. bei lösen der linearen least square Gleichungund
#  - Dimensonality Reduction
#
#  A = U·Σ·V^T
#
# wobei
# - A    = n×m Matrix mit reelen Wertenwelche zerlegt werden soll
#             => soll zerlegt werden
# - U    = m×m Matrix
#             => Spalten von U sind die Spalten-singular vectors von A
# - Σ    = diagonale m×n Matrix
#             => Werte sind singular values der Matrix A
# - V^T  = ist transponierte Matrix von n×n Matrix V
#             = > Spalten von V sind Reihen-singular vectors von A
#
## Zerlegung
from scipy.linalg import svd
A = array([
  [1, 2],
  [3, 4],
  [5, 6]])

U, s, V = svd(A)
print(U)
print(s)
print(V)

## Rekonstruktion
from numpy import zeros
# Sigma muss kompatibel für Multiplikation gemacht werden
# => benötigt, wenn m != n
Sigma = zeros((A.shape[0], A.shape[1]))
Sigma[:A.shape[1], :A.shape[1]] = diag(s)
# Rekonstruieren
B = U.dot(Sigma.dot(V))
print(B)

### Pseudoinverse
# Moore-Penrose Inverse
# Matrix-Inverse nur für quadratische Matrizen definiert
# => verallgemeinert Invertierung einer Matrix von n x n zu n x m (quadratisch => rechteckig)
# => erlaubt lösen von linearer least squares => mehr Beobachtungen als Variablen
#
#  A^+ = V · D^+ ·U^T
#
# wobei
# - A^+ = pseudoinverse,
# - D^+ = pseudoinverse der diagonal matrix Σ
# - V   = transponierte von V^T
#
# U und V durch SVD: A = U·Σ·V^T
# D^+ durch erzeugen einer diagonalen Matrix aus Σ - nehmen des inversen s_(1,1) => 1/s_(1,1)

## Mittels pinv()
from numpy.linalg import pinv
A = array([
  [0.1, 0.2],
  [0.3, 0.4],
  [0.5, 0.6],
  [0.7, 0.8]])
# Pseudoinverse berechnen
B = pinv(A)
print(B)
B.dot(A) # => Identitätsmatrix

## Mittels SVD
U, s, V = svd(A)
# reciprocals von s
d = 1.0 / s
# m x n Matrix D erstellen
D = zeros(A.shape)
# befüllen der n x n diagonal matrix D
D[:A.shape[1], :A.shape[1]] = diag(d)
# Pseudoinverse berechnen
B = V.T.dot(D.T).dot(U.T)
print(B)


## Dimensonality Reduction
# Geeignet, wenn mehr Spalten als Reihen da sind - z.b. bei Worthäufigkeiten
#
# 1. SVD auf original Daten anwenden
# 2. k größten singular values aus Σ auswählen.
# 3. Diese Spalten können aus Σ genommen werden und die Reihen aus V^T
#
# Approximation B an A kann dann konstruiert werden:
#
#    B = U· Σ_k ·V_k^T
#
# In NLP angewendet auf Matrizen mit Worthäufigkeiten/Frequenzen
#  => heisst dann Latent Semantic Analysis
# Man arbeitet dann mit beschreibenden Teildatensatz T
#  => ist dann Zusammenfassung / Projektion
#     T = U · Σ_k
# Transformation kann auch auf ganzem Datensatz berechnet und angewendet werden:
#     T = A · V_k^T
#
# Beispiel:
# - Mehr Spalten als Reihen
# - SVD wird berechnet und nur ersten zwei Spalten ausgewählt
# - dann wieder rekombiniert, um original Matrix zu approximieren
A = array([
  [1,2,3,4,5,6,7,8,9,10],
  [11,12,13,14,15,16,17,18,19,20],
  [21,22,23,24,25,26,27,28,29,30]])´

## Zerlegen
U, s, V = svd(A)
Sigma = zeros((A.shape[0], A.shape[1]))
Sigma[:A.shape[0], :A.shape[0]] = diag(s)
n_elements = 2
Sigma = Sigma[:, :n_elements]
V = V[:n_elements, :]

## Rekonstruieren
B = U.dot(Sigma.dot(V))
print(B)
# Transformieren
T = U.dot(Sigma)
print(T)
T = A.dot(V.T)
print(T)

## Sklearn stellt dies direkt mit TruncatedSVD bereit
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=2)
svd.fit(A)
# apply transform
result = svd.transform(A)
print(result)
