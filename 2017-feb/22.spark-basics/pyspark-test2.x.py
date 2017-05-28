from pyspark.sql import SparkSession

sparkSession1 = SparkSession.builder.master("local").appName("Test2").getOrCreate()
     
sparkSession2 = SparkSession.builder.master("local").appName("Test2").enableHiveSupport().getOrCreate()


