########## KAGGLE SNIPPETS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ggplot import *
import seaborn as sns
import xgboost as xgb
from sklearn import model_selection, preprocessing
color = sns.color_palette()
# für jeden chart die nächste farbe (nur target bekommt grün = color[1])


# Fehlende Daten anzeigen
data = meat
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# Missing data grafisch
missing_df = data.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots()
rects = ax.barh(ind, missing_df.missing_count.values, color='y')
_ = ax.set_yticks(ind)
_ = ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()


# Variable diskretisieren
_ = pd.cut(np.array([10,25,50,70,80]), [0,33,66,100], labels=['low','medium','high'])
mtcars['power'] = pd.cut(mtcars.hp, [0,70,200,1000], labels=['low','medium','high'])

# Outlier entfernen
def remove_outliers(data,cols = 'all',low=.05,high=.95,remove_rows=False):
    if cols != 'all':
        data = data[cols]
    quantiles_df = data.quantile([low, high])
    data = data.apply(lambda x: x[(x>=quantiles_df.loc[low,x.name]) &
                                    (x <= quantiles_df.loc[high,x.name])])
    if remove_rows:
        return data.dropna()
    else:
        return data

data1 = np.arange(0,10,1) + 1
data2 = [1,1,1,1,1,1,1,1,1,1]
data = pd.DataFrame({'col1':data1,'col2':data2})
cols = ['col1','cols2']
remove_outliers(data,['col1'],remove_rows=True)


# Date(-Count) Charts
# z.B. wieviel Produkte pro Stunde
meat["month"] = meat["date"].dt.month
cnt_srs = meat['month'].value_counts()

plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
plt.xticks(rotation='vertical')
plt.ylabel('Count', fontsize=12)
plt.xlabel('Month of the year', fontsize=12)
plt.title("Frequency distribution by days since prior order", fontsize=15)
plt.show()


# Heatmat
# Wenn eine (aggregierte) Variable, als die Funktion von zwei anderen dargestellt werden soll
# z.B. x = Hour-of-Day, y=Day-of-Week, => Färbung=Count-Of-Orders
#df = orders_df.groupby(["order_dow", "order_hour_of_day"])["order_number"].aggregate("count").reset_index()
#df = grouped_df.pivot('order_dow', 'order_hour_of_day', 'order_number')
data = sns.load_dataset("flights")
data = data.pivot("month", "year", "passengers")
# data.head() => guck dir die struktur an
sns.heatmap(data,center=flights.loc["January", 1955])
plt.show()


## ggplot
data = meat
date_col = ['date']
line_cols = ['beef','pork','turkey']
data[date_col] =  data[date_col] # pd.to_datetime(data[date_col])
data_tidy = pd.melt(data[date_col+line_cols], id_vars=date_col)
ggplot(aes(x='date', y='value', colour='variable'), data=data_tidy) + geom_line()
ggplot(aes(x='date', y='value', colour='variable'), data=data_tidy) + stat_smooth(span=0.10) + ggtitle("The Title")


# Feature Importance xg_boost
data = mtcars
leave_out = ['name','hp']
y_name =  'hp'
# power ist kategorische variable
for f in data.columns:
    if (data[f].dtype.name=='category'):
        data[f] = data[f].astype('object')
    if (data[f].dtype=='object'):
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(data[f].values))
        data[f] = lbl.transform(list(data[f].values))

train_y = data[y_name].values
train_X = data.drop(leave_out, axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1}
dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, height=0.8, ax=ax)
plt.show()


# Pie-Chart für Counts
data = mtcars
count_var = 'cyl'
plt.figure(figsize=(10,10))
pie_df = data[count_var].value_counts()
labels = (np.array(pie_df.index))  #  Value, welcher gecountet (hier cyl=2,4,8)
sizes = (np.array((pie_df / pie_df.sum())*100))
plt.pie(sizes, labels=labels,
        autopct='%1.1f%%', startangle=200)
plt.title("The title", fontsize=15)
plt.show()

# Barcharts für max/counts by groups
data = mtcars
mtcars.groupby('hp')['cyl'].max().head()
grouping = 'hp'
agg_col = 'cyl'
grouped_df = data.groupby(grouping)[agg_col].aggregate("max").reset_index()
grouped_df.head()
cnt_srs = grouped_df[agg_col].value_counts()
cnt_srs
##mtcars.head()
plt.figure()
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Number of products in the given order', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

# Barchart
plt.figure(figsize=(8,4))
int_level = mtcars['power'].value_counts()
_ = plt.figure(figsize=(8,4))
sns.barplot(int_level.index, int_level.values, alpha=0.8, color=color[1])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Power', fontsize=12)
plt.show()


#  Side-by-side Barchart
plt.figure(figsize=(8,4))
_ = plt.figure(figsize=(8,4))
sns.countplot(x='cyl', hue='power', data=mtcars)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Power', fontsize=12)
plt.show()

meat
ggplot(aes())
