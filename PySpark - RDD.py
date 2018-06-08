########## SPARK - RDD
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
#  - getypte Spalten
from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession,SQLContext
import pandas as pd
import numpy as np
import os
home = os.getenv("HOME")

#### SPARK INITIALISIEREN
# SparkContext: zum Erstellen der Verbindung
# (SparkConf enthält die Optionen für Verbindungsaufbau)
spark = (SparkConf()
         .setMaster("local")
         .setAppName("TestApp")
         .set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)


#### RDD API

### RDD erstellen
## Alternativ auch: hdfs://, s3a://
rdd1 = sc.parallelize([1,2,3,4,5,6])
flights = (sc
    .textFile('file:///',home,'/Documents/Data/flights_small.csv')
    .map(lambda x: x.split(","))
    .cache())
#HiveCtx = HiveContext(sc)
#rows = hiveCtx.sql('select name,age from users')


### RDD OPERATIONEN
# Zwei Arten von Aktionen
## TRANSFORMATIONS:
#   - Erzeugt ein neues RDD aus bestehendem RDD
#   - sind lazy
#     => bilden DAG bis eine Action ausgeführt wird
#        Knoten: RDD; Kante: Transformation
#   - Narrow Transformation:
#     + Kein Shuffling notwendig
#     + Alle Elemente um neue Elemente zu berechnen sind bereits in Partition des Parent-RDD
#     +  z.B. map(), filer()
#   - Wide Transformation:
#     + Shuffling notwendig
#     + benötige Elemente sind verschiedenen Parent-RDDs
#     + zB. groupbyKey(), reducebyKey()
#
# ACTIONS:
#   - Liefern echte Werte an Driver Programm zurück (vs liefert RDD)
#    => hierdurch Werte vom Executor zum Driver gesendet
#   - Erst an dieser Stelle wird der DAG zum DAG Scheduler geschickt
#     (dieser erstellt Stages von Tasks, welche parallel ausgeführt werden können)
#    => vorher passiert gar nichts
#  - z.B. reduce(), collect(), count(), countByValue(), take(), foreach(func)

# Einfache Action: Ersten drei "Reihen": Action take()
flights.take(3)

# Transformation + Action: Wieviel Flüge pro Origin
# Map: Nehm 11 Wert und Action: Wieviel pro Wert
flights.map(lambda x: x[11]).countByValue()

# Komplexer: Top 5 Origins:
(flights.map(lambda x: (x[11],1))
    .reduceByKey(lambda x,y: x+y)
    .map(lambda x: (x[1],x[0]))
    .sortByKey(ascending = False)
    .take(5))

# Filter
(flights
    .filter(lambda x: x[11] == 'LAX')
    .take(5))
