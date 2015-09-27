library(ggplot2)
library(Amelia)
library(stats)

# Step-1: Load the data

setwd("E:/data analytics/datasets/")
teens = read.csv("snsdata.csv", header = TRUE, na.strings=c("NA",""))

# Step-2: Explore the data
str(teens)
dim(teens)
head(teens)

# Step-3: Preprocess data/Feature Engineering

#Do we have missing data?
missmap(teens, main="Teen data from social network - Missings Map", 
        col=c("yellow", "black"), legend=FALSE)


#Analyze gender variable
table(teens$gender, useNA = "ifany")

#Handling missing data of gender variable
teens$female = ifelse(teens$gender == "F" & !is.na(teens$gender), 1, 0)
teens$no_gender = ifelse(is.na(teens$gender), 1, 0)
table(teens$gender, useNA = "ifany")
table(teens$female, useNA = "ifany")
table(teens$no_gender, useNA = "ifany")

missmap(teens, main="Teen data from social network - Missings Map", 
        col=c("yellow", "black"), legend=FALSE)

#Analyze age variable
summary(teens$age)
teens$age = ifelse(teens$age >= 13 & teens$age < 20,
                    teens$age, NA)
summary(teens$age)

#Handle the missing values of age variable
ave_age = ave(teens$age, teens$gradyear, FUN =
                 function(x) mean(x, na.rm = TRUE))
teens$age = ifelse(is.na(teens$age), ave_age, teens$age)
summary(teens$age)

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

# Step-5: Evaluating model performance
str(teen_clusters)

teen_clusters$size

teen_clusters$centers

teen_clusters$totss

teen_clusters$withinss

teen_clusters$tot.withinss

teen_clusters$betweenss

# Step-6: Validity of the model
teens$cluster = teen_clusters$cluster

teens[1:5, c("cluster", "gender", "age", "friends")]

aggregate(data = teens, age ~ cluster, mean)

aggregate(data = teens, female ~ cluster, mean)

aggregate(data = teens, friends ~ cluster, mean)


library(animation)

cent <- 1.5 * c(1, 1, -1, -1, 1, -1, 1, -1)
x <- NULL
for (i in 1:8) x <- c(x, rnorm(25, mean=cent[i]))
x <- matrix(x, ncol=2)
colnames(x) <- c("X1", "X2")
dim(x)

head(x)

par(mar=c(3, 3, 1, 1.5), mgp=c(1.5, 0.5, 0), bg="white")
kmeans.ani(x, centers=3, pch=1:4, col=1:4)
