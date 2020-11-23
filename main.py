import sys
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.types import *
from pyspark.sql import SparkSession


#start Spark Session
spark = SparkSession \
		.builder \
		.appName("Flag") \
		.getOrCreate()

print('************')
print('Python version: {}'.format(sys.version))
print('Spark version: {}'.format(spark.version))
print('************')


#schema
schema = StructType([
	StructField("BARS",IntegerType()),
	StructField("STRIPES",IntegerType()),
	StructField("COLOURS",IntegerType()),
	StructField("RED",IntegerType()),
	StructField("GREEN",IntegerType()),
	StructField("BLUE",IntegerType()),
	StructField("GOLD",IntegerType()),
	StructField("WHITE",IntegerType()),
	StructField("BLACK",IntegerType()),
	StructField("ORANGE",IntegerType()),
	StructField("MAINHUE",IntegerType()),
	StructField("CIRCLES",IntegerType()),
	StructField("CROSSES",IntegerType()),
	StructField("SALTIRES",IntegerType()),
	StructField("QUARTERS",IntegerType()),
	StructField("SUNSTARS",IntegerType()),
	StructField("CRESCENT",IntegerType()),
	StructField("TRIANGLE",IntegerType()),
	StructField("ICON",IntegerType()),
	StructField("ANIMATE",IntegerType()),
	StructField("TEXT",IntegerType()),
	StructField("BOTRIGHT",IntegerType())
	])


# Chargement de flag.cqsv

dataset = spark.read.csv("flag.csv", header = 'true', schema = schema)

# Save and load model
#clusters.save(sc, "model")

#Affichage de flag.csv
dataset.show()
dataset.printSchema()

#A feature transformer that merges multiple columns into a vector column.
vecAssembler = VectorAssembler(
		inputCol = ['BARS',
		'STRIPES',
		'COLOURS',
		'RED',
		'GREEN',
		'BLUE',
		'GOLD',
		'WHITE',
		'BLACK',
		'ORANGE',
		'MAINHUE',
		'CIRCLES',
		'CROSSES',
		'SALTIRES',
		'QUARTERS',
		'SUNSTARS',
		'CRESCENT',
		'TRIANGLE',
		'ICON',
		'ANIMATE',
		'TEXT',
		'BOTRIGHT'],
		outputCol = "features")

flag_with_features = vecAssembler.transform(dataset)

# Do K-means
k = 3 

kmeans_algo = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans_algo.fit(flag_with_features)
centers = model.clusterCenters()

# Assign clusters to flowers
flag_with_clusters = model.transform(flag_with_features)

print("Centers", centers)

# Convert Spark Data Frame to Pandas Data Frame
flag_for_viz = flag_with_clusters.toPandas()

# Vizualize
