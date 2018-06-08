########## NEURONALES NETZ
## Simuliert XOR Funktion mit 2 Inputs und 1 Output und drei Layers
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
#from sympy.integrals.risch import derivation
#from tensorflow.contrib.distributions.python.ops.bijectors import sigmoid

#### Sigmoid (und dessen Ableitung)
def sigmoid(x,deriv=False):
    if (deriv ==True): # wenn true, dann ableitung der sigmoid funktion = berechnung des fehlers für backpropagation
        return x*(1-x)
    return 1/ (1+np.exp(-x))  # sigmoid funktion


#### Initialisierung des INPUT als Matrix
# => jede Zeile eine Beobachtung
# => jede Spalte ein Neuron
# hier: 4 Trainingsbeispiele mit je 2 Inputs + Bias Term
# Erste Eingabe: 0,0,1 an den Neuronen
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

#### OUTPUT der XOR-Funktion
y= np.array([[0],
             [1],
             [1],
             [0]])

np.random.seed(1)


#### Gewichte
# => jedes Gewicht bekommt einen random-Wert zu<gewiesen
# Verbindungen Eingabe(3) -> Mittlerer-Layer(4)
syn0 = 2*np.random.random((3,4)) - 1 # 3 hoch x 4 breit Matrix mit [-1,1]
# Verbingen Mittel-Layer(4) -> Ausgabe(1)
syn1 = 2*np.random.random((4,1)) - 1 # 4 hoch x 1 breit Matrix mit [-1,1]


#### TRAINING
# Ausgabe zeigt die Anpassung des Fehlers zwischen dem Modell und dem gewünschten Ergebnis => verringert sich stetig
for j in range(200):
    l0 = X
    l1 = sigmoid(np.dot(l0,syn0))      # erster Prediction Step
                                       # => Matrix-Multiplikation zwischen jedem Layer und der Gewichte
                                       # => dann Anwendung der Sigmoid Funktion, um das nächste Layer zu erzeugen
                                       # enthält Prediction der Output-Daten
    l2 = sigmoid(np.dot(l1, syn1))     # dann nochmal dasselbe auf demnächsten Layer
                                       # dies ist eine verfeinerte Prediction

    logging.debug(str(j) + ": L2" + str(l2))
    l2_error = y - l2  # Vergleich des tatsächlichen Output mit dem erwarteten Ouput (durch Substraktion), um Error-Rate zu bekommen
    logging.debug(str(j) + ": L2-Error" + str(l2_error))

    # Ausgabe des Fehlers im festen Interval
    if (j % 10) == 0:
        logging.info ("ERROR at " + str(j) +  ": " + str(np.mean(np.abs((l2_error)))))

    ## ERROR Backpropagation
    l2_delta = l2_error * sigmoid(l2, deriv=True) # Multiplikation der Abweichung mit der Ableitung des Sigmoid-Funktion des Outputs von Layer 2
                                                  # Gibt delta, welches hilft den Fehler (bei jeder Iteration) zu verringern

    logging.debug("L1: " + str(l1))
    logging.debug (str(j) + ": L2-Delta" + str(l2_delta))
    l1_error = l2_delta.dot(syn1.T)               # Betrachten wieviel Layer 1 zu zum Fehler in Layer 2 beigetragen hat  => BACKPROPAGATION
                                                  # Durch Multiplikation des Delta von Layer 2 mit der transponierten Matrix von Layer 1
    logging.debug(str(j) + ": L1-Error" + str(l1_error))
    l1_delta = l1_error * sigmoid(l1,deriv=True)  # Delta des Layer 1 durch Mulitplikation der Fehlerrate von Layer 1 mit der Ableitung der Sigmoid Funktion (Ableitung von Layer 1)
    logging.debug(str(j) + ": L1-Delta" + str(l1_delta))

    # Da man nun die Deltas von jedem Layer kennt...
    # ANPASSEN DER Gewichte, um den Fehler bei jeder Iteration zu verringern => GRADIENT
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

    logging.debug("L2:" + str(l2))

logging.info("Output after training")
logging.info("\n" + str(l2))
