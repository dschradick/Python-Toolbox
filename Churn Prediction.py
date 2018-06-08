########## CHURN PREDICTION
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split

#### Daten laden
data = pd.read_csv("~/Documents/Data/turnover.csv")


#### Daten betrachten
data.head()
data.info()  # => einige müssen konvertiert werden
data.salary.unique()
data.department.unique()
data.mean()
data.describe()


#### Datentypen anpassen
## Object => Ordinale Variable (keine Dummies!)
# Ändern des Datentypen auf categorial
data.salary = data.salary.astype('category')
# Variable ordinal => Reihenfolge festlegen
data.salary = data.salary.cat.reorder_categories(['low','medium','high'])
data.salary.unique()
# Codierung der Kategorien durch Integers
data.salary = data.salary.cat.codes
data.salary.unique()

## Dummifizieren
departments = pd.get_dummies(data.department)
departments.columns
## Dummy-Trap vermeiden durch löschen einer (redundanten) Spalte
departments = departments.drop("technical", axis = 1)
# Alte Spalte 'department' im originalen df löschen
data = data.drop("department", axis=1)
# Dummifizierte Variable hinzufügen
data = data.join(departments)



#### Mini EDA
n_employees = len(data)
## Gegangen vs geblieben
# Anzahl geblieben / gegangen
data.churn.value_counts()
# Prozent geblieben / gegangen
data.churn.value_counts()/n_employees*100

## Korrelationen
corr_matrix = data.corr()
sns.heatmap(corr_matrix)
plt.show()



#### Churn Vorhersage
target = data.churn
features = data.drop("churn",axis=1)

## Train / Test-Split
target_train, target_test, features_train, features_test = train_test_split(target,features,test_size=0.25,random_state=0)

## Entscheidungsbaum verwenden
model = DecisionTreeClassifier(random_state=0)
model.fit(features_train,target_train)

## Scoring des Models
model.score(features_train,target_train)*100
model.score(features_test,target_test)*100

## Grafisches Modell exporteieren
export_graphviz(model,"tree.dot")



#### Evaluation

### Overfitting verhindern
# 1. Begrenzen der maximalen tree depth
#    => z.B. nur bis tiefe = 5 wachsen lassen
# 2. Begrenzen der minimalen sample size in Blättern
#    => z.B. darf nicht mehr wachsen, wenn nur noch 100 employees im Knoten
model_depth_5 = DecisionTreeClassifier(max_depth=5, random_state=0)
model_sample_100 = DecisionTreeClassifier(min_samples_leaf=100, random_state=0)
# => train und test performance betrachten

### Metrik wählen
# Wenn:
# Ziel churners => Fokus auf FN & Recall
# Recall = TP/(TP+FN)
#        = % der korrekten vorhersagen unter 1s (churners)
# => niedriger FN => höherer Recall
#
# Ziel churners: auch auf FP:
# Precision  = TP/(TP+FP)
#            = % der realen churners von predicted churners
# => nieriger FP, hohe precision?
#
# Ziel stayers => Fokus auf FP & Specificity
# Specificity = TN/(TN+FP)
#             = % der korrekten predictions unter 0s (stayers)
# => niedriger FP => höhere Specificity
from sklearn.metrics import precision_score, recall_score

prediction = model.predict(features_test)
precision_score(target_test, prediction)
recall_score(target_test, prediction)


## AUC
# Wenn nicht nur eine Metrik berücksichtigt werden soll,
# wie z.B. Recall oder Specificity, sondern beide
# Vertikal: Recall, Horizontal: 1 - Specificity
# Import the function to calculate ROC/AUC score
from sklearn.metrics import roc_auc_score

prediction = model.predict(features_test)
roc_auc_score(target_test, prediction)


#### Class Imbalances
## Prior verwenden
# => die wahrscheinlichkeiten in der gini-formel setzen
# class_weight="balanced"
# => verwendet Gewichte umgekehrt proportinal zur Klassenfrequenz der input data
# geht auch für einzelne observations:
# => sample_weight
# # http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

# Bei imbalanced
model_depth_7_u = DecisionTreeClassifier(max_depth=7,random_state=0)
model_depth_7_u.fit(features_train,target_train);
prediction_u = model_depth_7_u.predict(features_test)
recall_score(target_test,prediction_u)
roc_auc_score(target_test,prediction_u)
# Bei balanced
model_depth_7_b = DecisionTreeClassifier(max_depth=7,class_weight="balanced",random_state=0)
model_depth_7_b.fit(features_train,target_train);
prediction_b = model_depth_7_b.predict(features_test)
recall_score(target_test,prediction_b)  # recall höher
roc_auc_score(target_test,prediction_b)


#### Hyperparameter Tuning
## Cross-Validation vorher
from sklearn.model_selection import cross_val_score, GridSearchCV
cross_val_score(model,features,target,cv=10)
depth = [i for i in range(5,21)]
samples = [i for i in range(50,500,50)]
#weight = ['balanced','None']
parameters = dict(max_depth=depth,min_samples_leaf=samples)
## Grid Search
param_search = GridSearchCV(model, parameters)
param_search.fit(features_train, target_train)
print(param_search.best_params_)
# => min_samples_leaf = 50; max_depth = 5


#### Feature Importance
# berechnet als relativer decrease des Gini durch das selektierte Feature
feature_importances = model.feature_importances_
feature_list = list(features)
relative_importances = pd.DataFrame(index=feature_list, data=feature_importances, columns=["importance"])
relative_importances.sort_values(by="importance", ascending=False)


#### Feature Auswahl
# => nur Features mit Wichtigkeit > 1%
selected_features = relative_importances[relative_importances.importance>0.01]
selected_list = selected_features.index
features_selected = features[selected_list]


#### Finales Modell
# 1. mit vollen Daten
# 2. aber nur mit wichtigen Features
# 3. mit optimalen Parametern
final_model = DecisionTreeClassifier(min_samples_leaf = 50, max_depth = 5, random_state = 0)
final_model.fit(features_selected, target)
