########## ML - EINFACHER KLASSIFIKATOR
import pandas as pd
import matplotlib.pyplot as plt

#### Daten laden
data = pd.read_csv('~/Documents/Data/german_credit_test.csv')
target = 'Creditability'
X = data.drop(target,axis=1).iloc[:,1:]
y = data[target]


#### Modell erstellen
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0,stratify=y)
logreg.fit(X_train,y_train)


#### Vorhersage
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:,1]

#### Kurz Evaluation
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score

confusion_matrix(y_test,y_pred)
classification_report(y_test,y_pred)
cross_val_score(logreg,X,y,cv=5,scoring='roc_auc')

roc_auc_logit = roc_auc_score(y_test,y_pred_proba)
fpr_logit, tpr_logit, threshold = roc_curve(y_test, y_pred_proba)
plt.plot(fpr_logit, tpr_logit, 'b', label = 'AUC = %0.2f' % roc_auc_logit); plt.plot([0, 1], [0, 1],'r--'); plt.legend(loc=4);



#### WEITERES
### Entscheidungsbaum visualiseren
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
import pydot
from sklearn.externals.six import StringIO

model = DecisionTreeClassifier(random_state=0,max_depth=3)
model.fit(X_train,y_train)

## Scoring des Models
model.score(X_train,y_train)*100
model.score(X_test,y_test)*100

## Grafisches Modell exporteieren
dot_data = StringIO()
dot_data2 = export_graphviz(model,
    feature_names=X_train.columns,
    class_names=['no-default','default'], # ascending numerical ordr
    proportion=True, filled=True,
    rounded=True,  special_characters=True,
    out_file=dot_data) #None

# In Datei
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("graph.pdf")

# In Konsole (benötigt out_file=None)
#graph = graphviz.Source(dot_data2)
#graph





### SVM
from sklearn import svm
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

C_range = np.logspace(-2, 10, 4)
C_range
gamma_range = np.logspace(-9, 3, 4)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
grid = GridSearchCV(svm.SVC(probability=True), param_grid=param_grid, cv=cv)
grid.fit(X_train, y_train)
print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

y_pred = grid.predict(X_test)
y_pred_proba = grid.predict_proba(X_test)[:,1]
confusion_matrix(y_test,y_pred)
classification_report(y_test,y_pred)

roc_auc_svm = roc_auc_score(y_test,y_pred_proba)
fpr_svm, tpr_svm, threshold = roc_curve(y_test, y_pred_proba)
plt.plot(fpr_svm, tpr_svm, 'b', label = 'AUC = %0.2f' % roc_auc_svm); plt.plot([0, 1], [0, 1],'r--'); plt.legend(loc=4);



### LightGBM
import lightgbm as lgb
gbm = lgb.LGBMClassifier(objective='binary',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=30)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=10)


estimator = lgb.LGBMClassifier(objective='binary',num_leaves=31)
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [20, 50, 80] }

gbm = GridSearchCV(estimator, param_grid)
gbm.fit(X_train, y_train)

print('Best parameters: ', gbm.best_params_)
print('Best Score:', gbm.best_score_)
y_pred = gbm.predict(X_test)
y_pred_proba = gbm.predict_proba(X_test)[:,1]
confusion_matrix(y_test,y_pred)
classification_report(y_test,y_pred)

roc_auc_lgb = roc_auc_score(y_test,y_pred_proba)
fpr_lgb, tpr_lgb, threshold = roc_curve(y_test, y_pred_proba)
plt.plot(fpr_lgb, tpr_lgb, 'b', label = 'AUC = %0.2f' % roc_auc_lgb); plt.plot([0, 1], [0, 1],'r--'); plt.legend(loc=4);


### xgboost
from xgboost import XGBClassifier
xgb = XGBClassifier()

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
y_pred_proba = xgb.predict_proba(X_test)[:,1]
confusion_matrix(y_test,y_pred)
classification_report(y_test,y_pred)

roc_auc_xgb = roc_auc_score(y_test,y_pred_proba)
fpr_xgb, tpr_xgb, threshold = roc_curve(y_test, y_pred_proba)
plt.plot(fpr_xgb, tpr_xgb, 'b', label = 'XGB AUC = %0.2f' % roc_auc_xgb); plt.plot([0, 1], [0, 1],'r--'); plt.legend(loc=4);


### Vergleich
plt.plot(fpr_logit, tpr_logit, label = 'LogReg AUC = %0.3f' % roc_auc_logit)
plt.plot(fpr_svm, tpr_svm, label = 'SVM AUC = %0.3f' % roc_auc_svm)
plt.plot(fpr_lgb, tpr_lgb, label = 'LGB AUC = %0.3f' % roc_auc_lgb)
plt.plot(fpr_xgb, tpr_xgb, label = 'XGB AUC = %0.3f' % roc_auc_xgb)
plt.plot([0, 1], [0, 1],'r--'); plt.legend(loc=4);
