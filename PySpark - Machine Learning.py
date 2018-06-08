########## SPARK - MACHINE LEARNING
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
airports = spark.read.csv(home + '/Documents/Data/airports.csv',header=True, inferSchema = True)
flights = spark.read.csv(home + '/Documents/Data/flights_small.csv',header=True, inferSchema = True)
planes = spark.read.csv(home + '/Documents/Data/planes.csv',header=True, inferSchema = True)


#### PREPROCESSING
# benötigt numeric als typ
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

## Datentypen konvertieren
model_data = flights.join(planes, on="tailnum", how="leftouter")
model_data = model_data.withColumn("arr_delay", model_data.arr_delay.cast("integer"))
model_data = model_data.withColumn("air_time", model_data.air_time.cast("integer"))
model_data = model_data.withColumn("month", model_data.month.cast("integer"))
planes = planes.withColumnRenamed("year", "plane_year")
model_data = model_data.withColumn("plane_year", model_data.plane_year.cast("integer"))

## String Features konvertieren
carr_indexer = StringIndexer(inputCol="carrier", outputCol="carrier_index")
carr_encoder = OneHotEncoder(inputCol="carrier_index", outputCol="carrier_fact")
dest_indexer = StringIndexer(inputCol="dest", outputCol="dest_index")
dest_encoder = OneHotEncoder(inputCol="dest_index", outputCol="dest_fact")
vec_assembler = VectorAssembler(inputCols=["month", "air_time", "carrier_fact", "dest_fact", "plane_age"], outputCol="features")

## Feature Engineering
model_data = model_data.withColumn("plane_age", model_data.year - model_data.plane_year)
model_data = model_data.withColumn("is_late", model_data.arr_delay > 0)
model_data = model_data.withColumn("label", model_data.is_late.cast("integer"))
model_data = model_data.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")

## Pipeline bauen
flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])
piped_data = flights_pipe.fit(model_data).transform(model_data)
training, test = piped_data.randomSplit([.6, .4])



#### KLASSIFIKATION
from pyspark.ml.classification import LogisticRegression
import pyspark.ml.evaluation as evals
import pyspark.ml.tuning as tune


lr = LogisticRegression()
evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")
# Import the tuning submodule


grid = tune.ParamGridBuilder()
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])
grid = grid.build()
cv = tune.CrossValidator(estimator=lr,
                         estimatorParamMaps=grid,
                         evaluator=evaluator
                         )

best_lr = lr.fit(training)
print(best_lr)
test_results = best_lr.transform(test)
print(evaluator.evaluate(test_results))
