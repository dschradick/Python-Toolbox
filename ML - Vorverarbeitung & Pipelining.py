########## ML - VORVERARBEITUNG & PIPELINING
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
import pandas as pd
import numpy as np
from ggplot import diamonds
from bokeh.sampledata.autompg import autompg as auto


#### Daten vorbereiten
del auto['name']
X = diamonds.drop('price',axis=1)
X_numeric = X.select_dtypes(include='number')
y = diamonds.price




#### Dummies erzeugen
# Scikit-Learn konvertiert kategorische nicht per default zu dummies
# Funktionen
# - Sklearn: OneHotEncoder, LabelEncoder
# - Pandas: get_dummies()
### Pandas
## Zuerst Mapping für Spaltenbeschriftung
mapping = {1:'US',2:'Europe',3:'Asia'}
auto = auto.replace({'origin':mapping})
## Dummies erzeugen
auto2 = pd.get_dummies(auto,drop_first=True) # automatischer drop der redundaten Spalte
# oder: danach manuel del auto['origin_US']  # => wegen redundanter Information (problem bei einigen Algs)
## Sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
# LabelEncoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(auto.origin)
# OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoder.fit_transform(integer_encoded)
## Oder besser: direkt beides zusammen
label_binarizer = LabelBinarizer()
label_binarizer.fit_transform(auto.origin)

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]],drop_first=True)
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res)



#### Missing Values
# Strategien: mean, median, most_frequent
from sklearn.preprocessing import Imputer

x = np.array([1,2,3,np.NaN]).reshape(-1,1)
imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
imp.fit_transform(x)




### Scaling & Centering
# Oft haben Features verschiedene Skalen / Größenordnungen
# Problem: einige Algorithmen (z.B. KNN) verwenden euklidische Distanz als Information
#    => größere Skalen haben ein ungerechtfertigt großen Einfluss auf das Modell
# Lösung: Normalisierung
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

std_scaler = StandardScaler()
standardized_X = std_scaler.fit_transform(X_numeric)
mm_scaler = MinMaxScaler(feature_range=(0, 1))
min_max_X = mm_scaler.fit_transform(X_numeric)




#### Pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

## Einfach
imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
scaler = StandardScaler()

steps = [('imputation',imp),
         ('scaler',scaler)]

pipeline = Pipeline(steps)
pipeline.fit_transform(X_numeric)


## Einfache Pipeline
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
steps = [('imputation',imp),                       
         ('linear regression',linear_regression)]  

pipeline = Pipeline(steps)
X_train,X_test, y_train, y_test = train_test_split(X_numeric,y,test_size=0.3,random_state=42)
pipeline.fit(X_train, y_train);
y_pred = pipeline.predict(X_test)
pipeline.score(X_test,y_test)




## Mit verschiedenen Arten - numerisch + kategorisch - von Features
# FunctionTransformer erzeugt aus normaler Funktion ein Object, welches die Pipeline akzeptiert
# => erlaubt beliebige transformationen in der Pipeline auf die Daten anzuwenden
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion

get_numeric_data = FunctionTransformer(lambda x: x.select_dtypes(include='number'), validate=False)
mapper1 = DataFrameMapper([(['color'],[LabelBinarizer()])],sparse=False)
mapper2 = DataFrameMapper([(['cut'],[LabelBinarizer()])],sparse=False)
mapper3 = DataFrameMapper([(['clarity'],[LabelBinarizer()])],sparse=False)

process_and_join_features = FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('scaler', StandardScaler()),
                    ('imputer',  Imputer())
                ])),
                ('categorial_feature_1', Pipeline([('dummify_1', mapper1)])),
                ('categorial_feature_2', Pipeline([('dummify_2', mapper2)])),
                ('categorial_feature_3', Pipeline([('dummify_3', mapper3)]))
             ]
        )

pipeline = Pipeline([
        ('union', process_and_join_features),
        ('regressor', LinearRegression())
    ])

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
pipeline.fit(X_train, y_train);
y_pred = pipeline.predict(X_test)
pipeline.score(X_test,y_test)





#### Pipleine mit Gridsearch
# bei CV: cv = GridSearchCV(pipeline,param_grid=parameters),
# wobei hyperparamter mit stepname__hyperparamter=Value in den Parametern, danach cv.fit()

from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

steps = [('scaler', StandardScaler()),
         ('RidgeRegression', Ridge())]

#parameters = {'SVM__C':[1],'SVM__gamma':[0.01]}
parameters = {'RidgeRegression__alpha':[0.1, 1.0, 10.0]}

pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X_numeric,y,test_size=0.2,random_state=42)

cv = GridSearchCV(pipeline,param_grid=parameters)
cv.fit(X_train, y_train)
y_pred = cv.fit(X_train,y_train)
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print("Tuned Model Parameters: {}".format(cv.best_params_))
