library(ggplot2)
library(Amelia)
library(stats)

setwd("E://")
teens = read.csv("snsdata.csv", header = TRUE, na.strings=c("NA",""))

str(teens)
dim(teens)

teens1 = na.omit(teens)
dim(teens1)
str(teens1)

# Taking subset of features
interests = teens[5:40]

# Normalizing the variables sothat distance calculation is not biased
#interests_z = as.data.frame(lapply(interests, scale))
interests_z = scale(interests)

# Step-4: Build the model
#The high-school-age characters in general:
#a Brain, an Athlete, a Basket Case, a Princess, and a Criminal. 
set.seed(120)
teen_clusters = kmeans(interests_z, 5)

str(teen_clusters)

teen_clusters$size
teen_clusters$centers

teen_clusters$withinss
teen_clusters$betweenss

teen_clusters$totss
teen_clusters$tot.withinss

teens1$cluster = teen_clusters$cluster

teens[1:5, c("cluster", "gender", "age", "friends")]

aggregate(data = teens, age ~ cluster, mean)

aggregate(data = teens, female ~ cluster, mean)

aggregate(data = teens, friends ~ cluster, mean)

