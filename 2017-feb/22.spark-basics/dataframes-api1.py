from pyspark.sql import functions as F
from pyspark import SparkContext, SQLContext

sc = SparkContext("local", "Test")
print(sc)

sqlContext = SQLContext(sc)
sqlContext

#Creating data frame from list
data = [('John', 'Smith', 47),('Jane', 'Smith', 22), ('Frank', 'Jones', 28)]
schema = ['fname', 'lname', 'age']
df = sqlContext.createDataFrame(data, schema)
df

#Retrieving contents of data frame
df.printSchema()
df.show()
df.first()
df.count()

#Adding columns
df = df.withColumn('salary', F.lit(0))
df.show()
df.withColumn('salary2', df['age'] * 100).show()

#Filtering and subsetting 
df.filter(df['age'] > 30).select('fname','age').show()
df.select(F.max('age').alias('max-age')).show()

#Grouped aggregations
df.groupBy('lname').max('age').show()
