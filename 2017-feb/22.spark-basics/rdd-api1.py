from pyspark import SparkContext

sc = SparkContext("local", "RDD api")

data1 = [1,2,3,4,5,6,7,8,9,10]
rdd1 = sc.parallelize(data1,4)
print(type(sc))
print(type(data1))
print(type(rdd1))
rdd1.collect()

#map api
squared_rdd1 = rdd1.map(lambda x:x**2)
print(squared_rdd1.collect())

def square(x):
    return x**2

squared_rdd2 = rdd1.map(square)
print(squared_rdd2.collect())

#filter api
filtered_rdd = rdd1.filter(lambda x:x%2==0)
filtered_rdd.collect()

#flat map api
map_rdd1 = rdd1.map(lambda x:(x,x**3))
print(map_rdd1.collect())
flat_rdd2 = rdd1.flatMap(lambda x:(x,x**3))
print(flat_rdd2.collect())

#distinct api
data2 = [1,2,2,2,2,3,3,3,3,4,5,6,7,7,7,8,8,8,9,10]
rdd2 = sc.parallelize(data2,4)
distinct_rdd = rdd2.distinct()
print(distinct_rdd.collect())

data3 = [('Apple','Fruit',200),('Banana','Fruit',24),('Tomato','Fruit',56),('Potato','Vegetable',103),('Carrot','Vegetable',34)]
rdd3 = sc.parallelize(data3,4)

category_price_rdd = rdd3.map(lambda x: (x[1],x[2]))
category_total_price_rdd1 = category_price_rdd.reduceByKey(lambda x,y:x+y)
print(category_total_price_rdd1.collect())
category_total_price_rdd2 = category_price_rdd.groupByKey()
print(category_total_price_rdd2.collect())

rdd4 = sc.parallelize([1,2,3,4,5],4)
print(rdd4.count())
print(rdd4.take(2))

rdd5 = sc.parallelize([5,3,12,23])
print(rdd5.takeOrdered(3,lambda s:-1*s))

rdd6 = sc.parallelize([(5,23),(3,34),(12,344),(23,29)])
print(rdd6.takeOrdered(3,lambda s:-1*s[1]))