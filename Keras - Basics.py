########## KERAS - BASICS
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

### Versus Lineare Regression
# Bei Lineare Regression wird jedes Feature mit einem festen Wert versehen,
# welche den durchschnittlichen Einfluss auf die Zielvariable beschreibt.
# => Interaktionen (ausser explizit angegeben) werden nicht abgebildet
# Grafische Intution: Alle haben Feature Vektoren mit direktem Pfeil auf Zielvariable,
#                     aber keine Layer dazwischen die erst A und B zusammenführt.

### Layers
# Input Layer: Eingaben
# Output Layer: Ausgabe
# Hidden Layer: Während Input/Output Layer für die Welt sichtbar sind,
#               werden die Hidden Layer nicht direkt beobachtet
#   => 1. Jede Node im Hidden Layer repräsentiert eine Aggregation der Inputdaten und
#      2. Jede Node fügt eine weitere Möglichkeit hinzu die Interaktion zwischen den Knoten zu erfassen
#       => je mehr Nodes => je mehr Interaktion werden erfasst

### Forward Propagation
# Multiply - Add Prozess: Werte des vorigen Layers mit Gewichten multiplizieren
# und im ankommenden Konten addieren
#  => entspricht dem Dot-Produkt
# Jeweils für einen Datenpunkt, der durch die Werte in den Eingabe-Knoten codiert ist.

## Activation Function
# Funktion für den Output der Node - wird auf die Inputs der nodes angewendet
# früher: S-förmige Funktion tanh
# Heute meist: ReLU (Rectified Linear Activation)
# RELU(x) = 0 if x < 0
#           x if x >= 0
# => 0, wenn kleiner null, sonst den input

# Bsp: Kinder, Anzahl der Konten => Anzahl der Transaktionen
# Input Layer
input_data = np.array([2,3]) # 2 Kinder, 3 Konten
# Gewichte
weights ={'node_0': np.array([1,1]),'node_1': np.array([-1,1]),'output': np.array([2,-1]),}
# Hidden Layer berechnen
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = np.tanh(node_0_input)
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = np.tanh(node_1_input)
hidden_layer_values = np.array([node_0_output,node_1_output])
print('Hidden Layer: ', hidden_layer_values)
# Output berechnen
output = (hidden_layer_values * weights['output']).sum()
print('Output ', output) # Bsp: => im schnitt 1,2 Transaktionen

### Deep Neural Networks
# Bilden intern Repräsentationen von Mustern der Daten
# Nachfolgende Layers bilden immer komplexere Repräsentationen - z.B. 1. Linien 2. Quadarte, 3. Gesicht 4. konkrete Person
# => Modellierer muss die Interaktionen nicht spezifizieren
# Ersetzt zum Teil das Feature Engineering


## Optimierung
# Gewichte müssen so gewählt werden, dass die richtige Ausgabe rauskommt
# Ein Datenpunkt: kann dies leicht durch Betrachtung des Netzes gemacht werden
# Mehrere Datenpunkte - Cost-Funktion Mean Squared Error: Fehler (Abweichung von Zielwert) zu einer Zahl durch Addition(bei linear anstatt aktivierugnsfunktion) der quadratierten Fehler
#  Ziel:  Gewichte finden, welche den minimalen Fehler für die Cost-Funktion gibt
#  => Gradient Descent


## Gradient Descent:
# Zufälliger Startpunkt, Ableitung berechnen für Richtung, Schritt runter gehen, wiederholen, bis nicht mehr runtegeht
# Slope positiv: in entgegengesetzer Richtung der Slope = runtergehen zu niedrigen Zahlen
#  => Abziehen der Slope vom derzeitigen Gewicht
# Problem: Zu großer Schritt kann vom weg abführen
# Lösung: learning-rate  => update jedes Gewichts durch Substraktion learning-rate * slope
## Bsp: 3 --2--> 6   Zielwert: 10 , learning_rate = 0.01
# Zur Berechnung der Slope für die Gewichte muss multipliziert werden:
# 1. Steigung der Cost-Funktion bzgl des Werts der des Knotens in den es reingeht
# 2. Der Wert des Knoten der in das Gewicht führt
# 3. Steigung der Aktivierungsfunktion bzgl des Werts der eingeht (hier ohne Aktivierungsfunktion)
#    => 2 * -4 * 3 = -24 ; Neues Gewicht = 2 - 0.01 * (-24)

# Steigung berechnen und Gewichte updaten
gradient = 2 * input_data * error  # Slope (array of slopes nennt man gradient)
weights_updated = weights - learning_rate * gradient
preds_updated = (weights_updated * input_data).sum()
error_updated = preds_updated - target
print (error_updated)


### Backpropagation
# Berechnet die Gewichte sequentiell ausgehend vom Output (mittels gradient descent)
# Basiert auf Kettenregel in Analysis
## Vogehen
# 1. Start mit zufälligen Gewichten
# 2. Forward Progagation für erste prediction
# 3. Backward Propagation um die Slope der Cost-Funktion für jedes Gewicht zu berechnen
# 4. Multiplizieren des Slope mit der Learning-Rate und von den derzeitigen Gewichten Abziehen
# 5. Solange bis plateau

## Slope berechung: Gradienten für Gewichte sind Produkt von
# 1. Wert des eingehenden Knotens
# 2. Ableitung der Aktivierungsfunktion für den betrachteten Knoten
# 3. Ableitung des Cost-Funktion bzgl. des Output-Knotens

## Stochastic Gradient Descent (SGD)
# Für schnellere Berechung,
# Bei jedem Update der Gewichte nur die die Slope auf einer Teilmenge der Daten (batch) berechnen
# Nächstes Update ein anderen Batch verwenden, um es zu berchnen
# Sobald alle Daten verwendet worden sind => neu Anfangen
#   => Epoche: einmal durch alle Daten


#### KERAS
## Model bauen
# 1. Architektur spezifizieren: wieviele layers, wieviele Nodes in jedem Layer, Aktivierungsfunktion in den Layern
# 2. Compile: spezifieren der Lossfunktion und Optimierer(Learning-Rate)
# 3. Fit: Backpropagation
# 4. Prdiction
%cd ~
## Architektur
# 1a. Regression
data = pd.read_csv('~/Documents/Data/hourly_wages.csv',delimiter=',')
predictors = data.drop('wage_per_hour',axis=1)
target = data['wage_per_hour']
n_cols = predictors.shape[1] # => Anzahl der Knoten im Input-Layer
# 1b. Classification
data = pd.read_csv('data/categorial.csv')
predictors = data.drop(['result'],axis=1).as_matrix()
target = to_categorical(data.result)

# 2. Modell
model = Sequential()  # => nur kanten zu direkt nächstem Layer
model.add(Dense(100,activation='relu',input_shape=(n_cols,))) # nodes = 100,  number of predictive features is stored in n_cols, Dense = jede Node mit jeder der nächsten verbunden
model.add(Dense(100,activation='relu')) # 2. hidden layer
# 2a) Regression
model.add(Dense(1)) # 2. hidden layer
# 2b) Classification
model.add(Dense(100,activation='relu')) # 3. hidden layer
model.add(Dense(2,activation='softmax')) # Output-Layer: 2 bei categorial, wegen one-hot enconding (z.B. shot missed, shot made)

## Compiling
# z.B. interne Funktion erzeugen, um Backpropagation effizient zu machen
# 1. Optimierer spezfizieren: Kontrolliert die Lerning-Rate. z.B. Adam
# 2. Loss Function: "mean_squared_error" ist standard für regression-probleme, classfikation: 'categorial_crossentropy'

# a) Regression
model.compile(optimizer='adam',loss='mean_squared_error')
# b) Classification: (cc =log loss) - kleiner ist besser, metric=['accuracy'] = einfachere Diagnose,
#    Output-Layer muss softmax haben(damit Summe = 1 => Wahrscheinlichkeit angeben)
model.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['accuracy'])

## Fitting
# Anwenden von Backpropagation und Gradient Descent auf die Daten, um die Gewichte anzupassen
# Vorher: Scaling der Daten zuvor erleichtert Optimierung: Features haben im Schnitt einen ähnlichen hohen Wert
#   =  Wie: Substrahieren vom Feature das Means des Features und durch Standardabweichung geteilt
model.fit(predictors,target)

## Save,Load
model.save('Documents/Data/model.h5')
my_model = load_model('Documents/Data/model.h5')
my_model.summary()

## Prediction
from sklearn.metrics import r2_score
predictions = model.predict(predictors)
r2_score(target,predictions)
probability_true = predictions[:,1]


### Titanic
titanic_numeric = pd.read_csv('data/titanic_all_numeric.csv')

target = to_categorical(titanic_numeric.survived)
predictors = titanic_numeric[titanic.columns[1:]].values
n_cols = predictors.shape[1]

early_stopping_monitor = EarlyStopping(patience=2)
model = Sequential()
model.add(Dense(32,activation='relu',input_shape=(n_cols,)))
model.add(Dense(32,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(predictors,target,validation_split=0.3,epochs=20,callbacks = [early_stopping_monitor],verbose=False) # epochs kann mit Early Stopping auch hoch sein

predictions = model.predict(predictors[0:1])
predicted_prob_true = predictions[:,1]
print(predicted_prob_true)

# Plot Validation Scores
plt.plot(hist.history['val_loss'], 'r')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()


## Model Capacity
# = entspricht Fähigkeit des Netzes Muster in den Daten zu erkennen
#   -> mehr Kapazität, desto höhere Tendenz zu overfitting - zu wenig = underfitting
# =>  hängt eng mit Over/Underfitting zusammen
# Validation-Score ist DIE Metrik für die Qualität der zukünftigen Schätzung

## Vorgehen
# 1. Mit kleinem Netz anfangen => Validation-Score bekommen
# 2. Solange Kapzität hinzufügen bis Validation-Score nicht mehr besser wird
# Bsp: (1 Layer,100 Nodes) -> (1,250) -> (2,250) -> (3,250) -> (3,200)

#### Optimization
## Vorgehen: Gute Learning Rate bestimmen
n_cols = predictors.shape[1]
input_shape = (n_cols,)
def get_new_model(input_shape = input_shape):
    model = Sequential()
    model.add(Dense(100,activation='relu',input_shape=input_shape))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    return model

lr_to_test = [0.000001,0.01,1]
for lr in lr_to_test:
    model = get_new_model()
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer=my_optimizer,loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(predictors,target)

## Early Stopping
# Model trainiert solange bis keine Verbesserungen mehr kommen
from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience=2) # 2 Epochen ohne Verbesserung


# Selbst wenn Learning gut getuned, kann folgndes Probblem auftreten:
# Dying Neuron: Totes (Relu-)Neuron die nichts zum Modell beitragen, weil es unabhängig von Eingabe immer selben Wert ausgibt (wenn es null ist und immer negative inputs bekommt)
#   => schlechte Learning-Rate (oder andere Aktivierungsfunktion) => einfach dran denken, wenn Modell nicht richtig trainiert
# Vanishing Gradient Problem: z.B. bei Tanh, wenn mehrere Layers nur sehr geringe Slopes haben (weil sie auf flachem Teil der tanh-Kurve sind) => Updates durch Backprob fast 0


## MNIST
# 28x28  => 768 Punkte pro Bild
mnist = pd.read_csv('data/mnist.csv',header=None)
mnist.shape

target = mnist[0].values
predictors = mnist[mnist.columns[1:]].values
n_cols = predictors.shape[1]
n_cols
target = to_categorical(target)


early_stopping_monitor = EarlyStopping(patience=2)
model = Sequential()
model.add(Dense(50,activation='relu',input_shape=(784,)))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(predictors,target,validation_split=0.1,epochs=20,callbacks = [early_stopping_monitor],verbose=True) # epochs kann mit Early Stopping auch hoch sein

predictions = model.predict(predictors[0:1])
predicted_prob_true = predictions[:,2]
print(predicted_prob_true)
