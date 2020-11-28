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
		inputCols = ['BARS',
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
# Colors code k-means results, cluster numbers
colors = {0:'red', 1:'green', 2:'blue', 3:'orange'}

fig = plt.figure().gca(projection = '3d')
fig.scatter(flag_for_viz.BARS,
             flag_for_viz.STRIPES,
             flag_for_viz.RED,
             c = flag_for_viz.prediction.map(colors),
             marker = 's')

fig.scatter(flag_for_viz.GREEN,
             flag_for_viz.BLUE,
             flag_for_viz.GOLD,
             c = flag_for_viz.prediction.map(colors),
             marker = 's')

fig.scatter(flag_for_viz.WHITE,
             flag_for_viz.BLACK,
             flag_for_viz.ORANGE,
             c = flag_for_viz.prediction.map(colors),
             marker = 's')

fig.scatter(flag_for_viz.MAINHUE,
             flag_for_viz.CIRCLES,
             flag_for_viz.CROSSES,
             c = flag_for_viz.prediction.map(colors),
             marker = 's')

fig.scatter(flag_for_viz.SALTIRES,
             flag_for_viz.QUARTERS,
             flag_for_viz.SUNSTARS,
             c = flag_for_viz.prediction.map(colors),
             marker = 's')


fig.scatter(flag_for_viz.CRESCENT,
             flag_for_viz.TRIANGLE,
             flag_for_viz.ICON,
             c = flag_for_viz.prediction.map(colors),
             marker = 's')


fig.scatter(flag_for_viz.ANIMATE,
             flag_for_viz.TEXT,
             flag_for_viz.BOTRIGHT,
             c = flag_for_viz.prediction.map(colors),
             marker = 's')

fig.set_xlabel('BARS')
fig.set_ylabel('STRIPES')
fig.set_zlabel('BLUE')
plt.show()
