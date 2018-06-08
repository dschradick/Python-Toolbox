########## CLEANING MIT PANDAS
### Gute Grundform
# def cleaning_function(row_data):
#   # data cleaning / return
# df.apply(cleaning_function,axis=1)
# df.assert (df_column_data > 0).all()
import pandas as pd

stadt = ['München','Annaheim','Bern','Wohlheim','Widden','Berlin']
ten_k = [232,344,234,254,100,400]
thirty_k = [867,321,122,434,111,344]
fifty_k = [1000,532,552,353,543,1400]
labels = ['Stadt','>10K','>30k','>50k']
cols = [stadt,ten_k,thirty_k,fifty_k]
zipped = list(zip(labels,cols))
gehaelter = pd.DataFrame(dict(zipped))
gehaelter = gehaelter[['Stadt','>10K','>30k','>50k']]

# Grundform
def clean(value):
    # Return 1 if sex_value is 'Male'
    if value > 300:
        return value
    elif value <= 300:
        return value + 1
    else:
        return np.nan

# Clean die Spalte
gehaelter['>10K'] = gehaelter['>10K'].apply(clean)
#gehaelter['Stadt'] = gehaelter['Stadt'].apply(lambda x: x.replace('T', '')) Ts löschen
#gehaelter['Stadt'] = gehaelter['Stadt'].apply(lambda x: re.findall('[a-z]+', x)[0]) # den teil den man haben will


# Duplikate & Missing Values
gehaelter.drop_duplicates(inplace=True) # entfernt gleiche Zeilen
# Missing values, Datentypen ?
df[df == '?'] = np.nan # wie sehen die Missing-Values aus und erstzen
gehaelter.info() # -> Missing values?
gehaelter.dropna() # => entfernt reihen mit nan-werte (führt zu wenig daten)
gehaelter[['>10K','>30k','>50k']] = gehaelter[['>10K','>30k','>50k']].fillna(0) # mit Wert oder summary statistic wie median
gehaelter['>10K'] = gehaelter['>10K'].fillna(gehaelter['>10K'].mean())

# Daten-konvertieren / Umschreiben
categorize_label = lambda x: x.astype('category')
df[LABELS] = df[LABELS].apply(categorize_label,axis=0)

#besser: nan mit median von groupby füllen
by_sex_class = titanic.groupby(['sex','pclass'])
def impute_median(series):
    return series.fillna(series.median())
titanic.age = by_sex_class['age'].transform(impute_median)


assert pd.notnull(gehaelter).all().all() # erstes all() generiert colum mit true zweites nur true
assert (gehaelter > 0).all().all() # alle Werte größer 0

# Datentypen fixen
df.dtypes # dtypes anzeigen
gehaelter['>10K'] = pd.to_numeric(gehaelter['>10K'],errors='coerce') # # Int-Konvertierung
gehaelter['Stadt'] = gehaelter['Stadt'].astype(str) # Konvertiere nach Strings
#gehaelter['Stadt'] = gehaelter['Stadt'].astype('category') # Konvertiere nach Kategorie => schneller
assert gehaelter['>10K'].dtypes == np.int64
assert gehaelter['Stadt'].dtypes == np.object

# Format fixen
gehaelter['>10K'].apply(lambda x:'{:0>5}'.format(x))

# Invalide Einträge suchen
staedte = gehaelter['Stadt']
staedte = staedte.drop_duplicates()
pattern = '^[A-Za-z\.\s]*$' # Erlaubt: Klein/Grossbuchstaben,Punkte, Leerzeichen
#Andere '\d{3}-\d{3}-\d{4}' => 123-456-7890 ;  '\$\d{3}.\d{2}' => $123.45 ; '[A-Z]\w*' => Australia
boolean_mask = staedte.str.contains(pattern)
mask_inverse = ~boolean_mask
invalide_staedte = staedte[mask_inverse]

# String Operationen
gehaelter['e_split'] = gehaelter.Stadt.str.split("e") # str
gehaelter['after_e'] = gehaelter.e_split.str.get(0) # zugriff auf array
gehaelter = gehaelter.drop(['e_split','after_e'], 1) # Spalte löschen ; 1 = axis
import re
matches = re.findall('\d+', 'Stadt 4 ist schöner als 33') # => alle Zahlen finden und als Liste zurück
matches = re.findall('\d{2}-\d{2}-\d{4}','Es ist Jahr 20-05-2017 heute') # => alle Datums

print(gehaelter) #=> drei vars = Stadt,Gehalt,Häufigkeit des Gehalts - hier: Wert (Gehalt) im Spaltennamen

# Reshaping Data - Melt & Pivot
# Melting: tranformiert Spalten in Reihe
# Pivoting: transformiert unique Werte in eigene Spalte
# => tranformiert "analyse-darstellung" nach "reporting-darstellung"
# Melting: löst Problem wenn die Spaltennamen jeweils den Wert beinhalten vs die Variable
gehaelter_melt = pd.melt(frame=gehaelter, id_vars =['Stadt'],
    value_vars=['>10K','>30k','>50k'], var_name='Gehalt', value_name='Häufigkeit')
gehaelter_melt
# Pivoting - verwandet eindeutige Werte in eigene Spalte
# (Gegenteil von Melting => nimmt bestimmten Wert und macht daraus eine Spalte)
gehaelter_melt.pivot(values='Häufigkeit',index='Stadt',columns='Gehalt')
# => Problem, bei doppelten Datenwerten => Pivot-Tabelle
# Privot-Tabelle - erlaubt Parameter - die Aggregationsfunktion - der angibt wie mit mehrfach vorkommenden Werten umzugehen ist
gehaelter_melt.pivot_table(values='Häufigkeit',index='Stadt',columns='Gehalt',aggfunc=np.mean,margins=True)
# (danach .reset_index(), um hierachical/MultiIndex nach RangeIndex )
# Index pivoten mit stack & unstack und drehen mit swaplevel(0,1), danach .sort_index()

gehaelter
# Merge
stadt = ['München','Annaheim','Bern','Wohlheim','Widden','Berlin']
bewohner = [1000,1231,4242,2534,1030,4400]
labels = ['Stadt','Einwohner']
cols = [stadt,bewohner]
zipped = list(zip(labels,cols))
einwohner = pd.DataFrame(dict(zipped))

join_table = pd.merge(left=gehaelter, right=einwohner, left_on='Stadt', right_on='Stadt')
# n_m_join = pd.merge(left=join_table, right=third_table,
