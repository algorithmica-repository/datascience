from pyspark import SparkContext

sc = SparkContext("local", "RDD api")

lines = sc.textFile("/home/algo/train.csv") 
print(type(lines))
print(lines.count())
print(lines.collect())


