from pyspark import SparkContext, SQLContext

sc = SparkContext("local", "Test")
print(sc)

sqlContext = SQLContext(sc)
sqlContext

