########## ML - KLASSIFIKATOREN EVALUATION
import pandas as pd


#### Daten laden
data = pd.read_csv('~/Documents/Data/german_credit_test.csv')
target = 'Creditability'
X = data.drop(target,axis=1)
y = data[target]



#### Einfaches Modell
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0,stratify=y)
logreg.fit(X_train,y_train)




#### Evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, cohen_kappa_score
from sklearn.model_selection import cross_val_score

y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:,1]

## Cross-Validation
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, random_state=0)
# roc_auc, f1, precision - http://scikit-learn.org/stable/modules/model_evaluation.html
scoring = 'accuracy'
results = cross_val_score(logreg, X_train, y_train, scoring=scoring, cv=kfold)
print(results.mean())

## Konfusions-Matrix
conf_matrix = confusion_matrix(y_test,y_pred)
print('Confusion-Matrix: \n{}'.format(conf_matrix))

## Klassifkations-Bericht
class_report = classification_report(y_test,y_pred)
print('Classification-Report: \n{}'.format(class_report))

## Einzelne Metriken
accuracy_score(y_test,y_pred)
cohen_kappa_score(y_test,y_pred)
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)


## AUC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
score = roc_auc_score(y_test,y_pred_proba)
cv_auc = cross_val_score(logreg,X,y,cv=5,scoring='roc_auc')
print('AUC: {}'.format(score))
print("CV AUC scores:\n {}".format(cv_auc))



## ROC Kurve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
fpr,tpr,thresholds = roc_curve(y_test,y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='AUC = %0.2f' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC-Curve')
plt.legend(loc="lower right")
plt.show()



## Precision-Recall Kurve
# wenn Klassen sehr unbalanciert
from sklearn.metrics import precision_recall_curve,  average_precision_score
average_precision = average_precision_score(y_test, y_pred_proba)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))




## Mehrere Klassifikatoren vergleichen mit ROC-Kurve
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:,1]
fpr_rf, tpr_rf, thresholds = roc_curve(y_test,y_pred_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train,y_train)
y_pred_lda = lda.predict(X_test)
y_pred_proba_lda = lda.predict_proba(X_test)[:,1]
fpr_lda,tpr_lda,thresholds = roc_curve(y_test,y_pred_proba_lda)
roc_auc_lda = auc(fpr_lda, tpr_lda)



plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='Logistic Regression AUC = %0.2f' % roc_auc)
plt.plot(fpr_lda,tpr_lda,label='Discriminat Analyis AUC = %0.2f' % roc_auc_lda)
plt.plot(fpr_rf,tpr_rf,label='Random Forest AUC = %0.2f' % roc_auc_rf)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-Curve')
plt.legend(loc="lower right")
plt.show()
