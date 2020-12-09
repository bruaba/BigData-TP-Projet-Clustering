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
	StructField("COUNTRY",StringType()),
	StructField("LANDMASS",IntegerType()),
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
		inputCols = [
		'BARS',
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
k = 6

kmeans_algo = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans_algo.fit(flag_with_features)
centers = model.clusterCenters()

# Assign clusters to flowers
flag_with_clusters = model.transform(flag_with_features)

print("Centers", centers)

# Convert Spark Data Frame to Pandas Data Frame
flag_for_viz = flag_with_clusters.toPandas()

# Vizualize
# Marker styles are calculated from LANDMASS
print(flag_for_viz['COUNTRY'] )


NAmericaI = flag_for_viz['LANDMASS'] == 1
NAmerica = flag_for_viz [ NAmericaI ]
SAmericaI = flag_for_viz['LANDMASS'] == 2
SAmerica = flag_for_viz [ SAmericaI ]
EuropeI = flag_for_viz['LANDMASS'] == 3
Europe = flag_for_viz [ EuropeI ]
AfricaI = flag_for_viz['LANDMASS'] == 4
Africa = flag_for_viz [ AfricaI ]
AsiaI = flag_for_viz['LANDMASS'] == 5
Asia = flag_for_viz [ AsiaI ]
OceaniaI = flag_for_viz['LANDMASS'] == 6
Oceania = flag_for_viz [ OceaniaI ]


# Colors code k-means results, cluster numbers



# Dimenstion reduction. From 22D to 3D
# by PCA method
datamatrix =  RowMatrix(dataset.select([
		'BARS',
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
		'BOTRIGHT'

	]).rdd.map(list))

# Compute the top 3 principal components. The "best" hyperplane.
pc = datamatrix.computePrincipalComponents(3)
print ("***** 3 Principal components *****")
print(pc)

# project data
projected = datamatrix.multiply(pc)
new_X = pd.DataFrame(
    projected.rows.map(lambda x: x.values[0]).collect()
)
new_Y = pd.DataFrame(
    projected.rows.map(lambda x: x.values[1]).collect()
)
new_Z = pd.DataFrame(
    projected.rows.map(lambda x: x.values[2]).collect()
)

# Vizualize with PCA, 3 components
# Colors code k-means results, cluster numbers
colors = {0:'green', 1:'yellow', 2:'red', 3:'blue', 4:'purple', 5:'black' }

fig = plt.figure().gca(projection='3d')
fig.scatter(new_X [NAmericaI],
            new_Y [NAmericaI],
            new_Z [NAmericaI],
            c = NAmerica.prediction.map(colors),
            marker = 's')
fig.scatter(new_X [SAmericaI],
            new_Y [SAmericaI],
            new_Z [SAmericaI],
            c = SAmerica.prediction.map(colors),
            marker = 's')

fig.scatter(new_X [EuropeI],
            new_Y [EuropeI],
            new_Z [EuropeI],
            c = Europe.prediction.map(colors),
            marker = 's')
fig.scatter(new_X [AfricaI],
            new_Y [AfricaI],
            new_Z [AfricaI],
            c = Africa.prediction.map(colors),
            marker = 's')
fig.scatter(new_X [AsiaI],
            new_Y [AsiaI],
            new_Z [AsiaI],
            c = Asia.prediction.map(colors),
            marker = 's')
fig.scatter(new_X [OceaniaI],
            new_Y [OceaniaI],
            new_Z [OceaniaI],
            c = Oceania.prediction.map(colors),
            marker = 's')


for i in range( len(flag_for_viz['COUNTRY'])): 
	fig.text (float(new_X.iloc[i]), float(new_Y.iloc[i]), float(new_Z.iloc[i]), flag_for_viz['COUNTRY'].iloc[i])

fig.set_xlabel('Component 1')
fig.set_ylabel('Component 2')
fig.set_zlabel('Component 3')
plt.savefig("plot_3D.png")



#2D

# Dimenstion reduction. From 22D to 2D
# by PCA method
datamatrix =  RowMatrix(dataset.select([
		'BARS',
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
		'BOTRIGHT'

	]).rdd.map(list))

# Compute the top 2 principal components. The "best" hyperplane.
pc = datamatrix.computePrincipalComponents(2)
print ("***** 2 Principal components *****")
print(pc)

# project data
projected = datamatrix.multiply(pc)
new_X = pd.DataFrame(
    projected.rows.map(lambda x: x.values[0]).collect()
)
new_Y = pd.DataFrame(
    projected.rows.map(lambda x: x.values[1]).collect()
)
# Vizualize with PCA, 2 components
# Colors code k-means results, cluster numbers
colors = {0:'green', 1:'yellow', 2:'red', 3:'blue', 4:'purple', 5:'black' }

fig = plt.figure().gca()
fig.scatter(new_X [NAmericaI],
            new_Y [NAmericaI],
            
            c = NAmerica.prediction.map(colors),
            marker = 's')
fig.scatter(new_X [SAmericaI],
            new_Y [SAmericaI],
            
            c = SAmerica.prediction.map(colors),
            marker = 's')

fig.scatter(new_X [EuropeI],
            new_Y [EuropeI],
                       c = Europe.prediction.map(colors),
            marker = 's')
fig.scatter(new_X [AfricaI],
            new_Y [AfricaI],
                       c = Africa.prediction.map(colors),
            marker = 's')
fig.scatter(new_X [AsiaI],
            new_Y [AsiaI],
                     c = Asia.prediction.map(colors),
            marker = 's')
fig.scatter(new_X [OceaniaI],
            new_Y [OceaniaI],
                        c = Oceania.prediction.map(colors),
            marker = 's')

for i in range( len(flag_for_viz['COUNTRY'])): 
	fig.text (float(new_X.iloc[i]), float(new_Y.iloc[i]), flag_for_viz['COUNTRY'].iloc[i])


fig.set_xlabel('Component 1')
fig.set_ylabel('Component 2')
plt.savefig("plot_2D.png")



"""
alias python="python3"
export PYSPARK_PYTHON=python3
"""