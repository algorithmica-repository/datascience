from pyspark import SparkContext, SQLContext
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml.regression import GBTRegressor

sc = SparkContext("local", "Test")
sqlContext = SQLContext(sc)
sqlContext


file = "/home/algo/hour.csv"
df = sqlContext.read.format('csv').option("header", 'true').load(file)
df.cache()
df.show()
df.printSchema()
df.count()
df.first()

df = df.drop("instant").drop("dteday").drop("casual").drop("registered")
df.printSchema()
df.columns

from pyspark.sql.functions import col
df = df.select([col(c).cast("double").alias(c) for c in df.columns])
df.printSchema()

train, test = df.randomSplit([0.7, 0.3])
print(train.count())
print(test.count())

#preprocessing stage
featuresCols = df.columns
featuresCols.remove('cnt')
# This concatenates all feature columns into a single feature vector in a new column "rawFeatures".
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")
# This identifies categorical features and indexes them.
vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=4)

# Takes the "features" column and learns to predict "cnt"
gbt = GBTRegressor(labelCol="cnt")


# Define a grid of hyperparameters to test:
#  - maxDepth: max depth of each decision tree in the GBT ensemble
#  - maxIter: iterations, i.e., number of trees in each GBT ensemble
# In this example notebook, we keep these values small.  In practice, to get the highest accuracy, you would likely want to try deeper trees (10 or higher) and more trees in the ensemble (>100).
paramGrid = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [2, 5])\
  .addGrid(gbt.maxIter, [10, 100])\
  .build()
# We define an evaluation metric.  This tells CrossValidator how well we are doing by comparing the true labels with predictions.
evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())
# Declare the CrossValidator, which runs model tuning for us.
cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid)

pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, cv])

pipelineModel = pipeline.fit(train)

pipelineModel.transform(test)\
    .select("features", "cnt", "prediction")\
    .show()
