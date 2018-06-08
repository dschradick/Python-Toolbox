########## SPARK - DF & SQL
### Datenstrukturen
# RDD = Resilient Distributed Dataset
#  - zugrundeliegende Datenstruktur
#  - ermöglicht Verteilung der Daten und der Berechnung
#    => sehr rudimentär
# Dataframe
#  - ist Dataset welches in benannte Spalten organisiert ist
#   =>  DataFrame = Dataset von Rows
#   =>  KEINE Datentypen => untyped transformations”
#  - Tabelle mit benanntenn Spalten
#    => konzeptionell wie Tabelle in Datenbank
#  - struktur erlaubt bessere optimierung
#    => benutzt catalyst query optizimer
# Dataset
#  - getypte Spalten (Scala)
from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession,SQLContext
import pyspark.sql.functions as fn
import pandas as pd
import numpy as np
import os
from ggplot import mtcars as mc
import seaborn as sns


home = os.getenv("HOME")

#### SPARK INITIALISIEREN
conf = (SparkConf()
         .setMaster("local")
         .setAppName("TestApp")
         .set("spark.executor.memory", "1g"))

sc = SparkContext(conf = conf)
sqlc = SQLContext(sc)
spark = SparkSession.builder.getOrCreate()



### DATEN EINLESEN
file_path = home + '/Documents/Data/flights_small.csv'
airports = spark.read.csv(file_path,header=True, inferSchema = True)
# df = spark.read.json("customer.json")
# df = spark.read.load("users.parquet")


#### DATEN HOLEN
result = mtcars.select('mpg').collect()
for x in result:
    print(x[0])


### TABELLEN ERZEUGEN
mtcars = spark.createDataFrame(mc)
mtcars.createTempView('mtcars')                             # Nur für die Session
mtcars.createGlobalTempView('mtcars')                       # Geteilt für mehrere sessions
mtcars.createOrReplaceTempView("mtcars3")
sqlc.dropTempTable("mtcars3")                               # Temporäre Tabelle löschen
spark.catalog.listTables()                                  # Tabellen anzeigen



#### PARTIONIEREN
# repartition() = vollständiger shuffle - erzeugen Partionen von gleicher Größe
# coalesce()    = kombiniert lediglich Partitionen
mtcars.repartition(10)                     # erhöht/verringert Anzahl der Partitionen
mtcars.rdd.getNumPartitions()              # Anzahl der Partionen
mtcars.coalesce(10)                        # verringert die Anzahl der Partitionen



#### CACHING & PERSISTENCE
# .cache()   = speichern mit default storage-level - MEMORY_ONLY
# .persist() = speichern spezifiertem storage-level
#             (MEMORY_ONLY,MEMORY_AND_DISK, DISK_ONLY)
data1 = mtcars.filter('hp > 120').cache()
data2 = mtcars.filter('hp > 120').persist()
data2.unpersist()



#### DATEN BESCHREIBEN
mtcars.show()                              # Daten betrachten
mtcars.printSchema()                       # Datentypen
mtcars.explain()                           # Datentypen + Physical Plan
mtcars.describe().show()                   # Statistiken
mtcars.crosstab('cyl','am').show()         # Frequenztabelle
mtcars.corr('hp','cyl')                    # Korrelation


#### VSISUALISIEREN
# Wenn kleines Result - gruppieren nachher wegen einfachem pandas plotting
# Wenn großes Result  - in Spark berechnen und dann seaborn
(mtcars
    .where('hp > 100')
    .toPandas()
    .groupby(['cyl','am'])['hp']
        .mean().unstack()
    .plot(kind='barh'))
data = (mtcars
    .where('hp > 100')
    .groupby('cyl','am').agg(fn.mean('hp').alias('hp')).toPandas())
data = spark.sql('select cyl,am,avg(hp) as hp from mtcars group by cyl,am').toPandas()
sns.barplot(x='hp',y='cyl',hue='am',orient='h',data=data)

#### SPALTEN
# Dataframe ist immutable
# => kann im gegensatz zu Pandas nicht geändert werden
# => alle manipulierenden Methoden liefern einen neuen Dataframe
mtcars.select('mpg').show()                                # Auswahl
mtcars.select('mpg','hp').show()
df = df.drop("address", "phoneNumber")                     # Spalte löschen
m2 = mtcars.withColumn('hp2',mtcars.hp*2)                  # Neue Spalte



#### BEOBACHTUNGEN
mtcars.head(5)
mtcars.head(5).toPandas()                                  # Resultat nach Pandas
mtcars.count()
mtcars.distinct().count()                                  # count(distinct)
mtcars.sample(withReplacement=True, fraction=0.1, seed=0)  # Stichprobe
## SQL
query = "SELECT MAX(mpg) FROM mtcars\
         GROUP BY cyl \
            HAVING mean(hp) > 100 \
         LIMIT 1"
spark.sql(query).show()
spark.sql(query).toPandas()
query = "SELECT COUNT(*) AS N, \
                AVG(hp) AS AHP, \
         CASE WHEN 'AM'=1 \
            THEN 'auto' \
            ELSE 'manual'\
         END AS SHIFT \
         FROM mtcars \
         GROUP BY cyl"
query
spark.sql(query).toPandas()
## QUERY
mtcars.filter('hp > 120').show()                           # WHERE
mtcars.where('hp > 120').show()                            # =...
mtcars.filter('cyl in (6,8)').show()                       # IN
mtcars.filter(mtcars.name.startswith('Maz')).show()        # String Filter
mtcars.groupby('cyl').agg(fn.mean('hp').alias('ps')).show()# GROUP BY
(mtcars.select(                                            # CASE/WHEN
    'hp', fn.when(mtcars.hp > 200, 'big')
    .otherwise("small")).show())



#### DATA CLEANING
# => immer mtcars = mtcars... => nicht inplace
mtcars = mtcars.dropDuplicates()             # Duplikate löschen
mtcars.na.drop().show()                      # NAs droppen
mtcars.na.fill(50).show()                    # NAs ersetzen
mtcars.na.replace(10, 20).show()             # Ersetzen von Wert


#### JOINING
print(airports.show())
airports = airports.withColumnRenamed("faa", "dest")
flights_with_airports = flights.join(airports, on="dest", how="leftouter")
print(flights_with_airports.show())



planes = planes.withColumnRenamed("year", "plane_year")
model_data = flights.join(planes, on="tailnum", how="leftouter")
