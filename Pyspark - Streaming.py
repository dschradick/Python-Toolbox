########## PYSPARK - SPARK STREAMING
# Normales Spark Streaming verwendet DStream API
#  => unterliegend: einfache RDDs => nur map(),reduce(),...
# Structured Streaming verwendet Dataset API
#  => dort wachsende Tabelle mit benannten Spalten

from pyspark import SparkContext
from pyspark.streaming import StreamingContext


#### SPARK STREAMING INITIALISIEREN
sc = SparkContext("local[*]", "StreamWordCounter")
ssc = StreamingContext(sc, 10)
## Verzeichnis für die Checkpoints
ssc.checkpoint("/tmp")


#### INPUTSTREAM
# Netzwerksocket als Inputstream
# für Datei: ssc.textFileStream()
text = ssc.socketTextStream("localhost", 9999)


#### STREAM BEARBEITUNG
# Update-Funktion für den Count
def updateTotalCount(currentCount, countState):
    if countState is None:
       countState = 0
    return sum(currentCount, countState)

### Wörter zählen
# Für jede Zeile im Micro-Batch: Split und erzeuge List von Wörtern
countStream = text.flatMap(lambda line: line.split(" "))\
                   .map(lambda word: (word, 1))\
                   .reduceByKey(lambda a, b: a + b)

# update total count for each key
totalCounts = countStream.updateStateByKey(updateTotalCount)
totalCounts.pprint()



#### STREAMING STARTEN
ssc.start()
ssc.awaitTermination()
