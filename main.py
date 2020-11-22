from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession
from pyspark import SparkContext

spark = SparkSession \
		.builder \
		.appName("PythonSparkProjet") \
		.getOrCreate()

# Chargement de flag
sc = SparkContext("local")
dataset = sc.textFile("flag.csv")
parsedData = dataset.map(lambda line: array([ float(x) for x in line.split(",")]))

clusters = KMeans.train(parsedData, 2, maxIterations = 10, initializationMode = "random")

def error(point):
	center = clusters.centers[clusters.predict(point)]
	return sqrt( sum( [x**2 for x in (point - center)]))

WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

# Save and load model
clusters.save(sc, "model")

#Affichage de flag.csv
dataset.show()
dataset.printSchema()