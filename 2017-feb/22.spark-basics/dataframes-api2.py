from pyspark import SparkContext, SQLContext

sc = SparkContext("local", "Test")
sqlContext = SQLContext(sc)
sqlContext


file = "/home/algo/Downloads/bike-sharing/hour.csv"
df = sqlContext.read.format('csv').option("header", 'true').load(file)
df.cache()
df.show()
df.printSchema()
df.count()
df.first()
