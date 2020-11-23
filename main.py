import sys
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark import SparkContext


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