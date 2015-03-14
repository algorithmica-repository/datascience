# load required libraries
library(dplyr)

# loading dataset
Forest = read.csv("E:/data analytics/datasets/forestfires.csv", header = TRUE, sep = ',')
Forest

# Task 1 

# How many observations are there in the dataset?
nrow(Forest)

Forest %>% summarize(count=n())

# How many observations are there with a fire (i.e. area>0)?
nrow(Forest[Forest$area>0, ])

nrow(subset(Forest, area>0))

Forest %>% filter(area>0) %>% summarise(count=n())


# How many observations are there with rain (i.e. rain>0)?
nrow(Forest[Forest$rain>0, ])

Forest %>% filter(rain>0) %>% summarise(count=n())

# How many observations are there with both a fire and rain?
nrow(Forest[Forest$area>0 & Forest$rain>0, ])

Forest %>% filter(area>0 & rain>0) %>% summarise(count=n())


# Task 2

# Show the columns month, day, area of all the observations.
Forest[, c('month', 'day', 'area')]

Forest %>% select(month, day, area)

# Show the columns month, day, area of the observations with a fire.
Forest[Forest$area>0, c('month', 'day', 'area')]

Forest %>% filter(area>0) %>% select(month, day, area)


# Task 3

# How large are the five largest fires (i.e. having largest area)? 
Forest %>% arrange(desc(area)) %>% top_n(5) %>% select(area)

# What are the corresponding month, temp, RH, wind, rain, area?
Forest %>% arrange(desc(area)) %>% top_n(5) %>% select(month, temp, RH, wind, rain, area)


# Task 4

# Reorder factor levels of month to be from Jan to Dec.
levels(Forest$month)
Forest$month = factor(Forest$month, levels = c('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'))
levels(Forest$month)

# Add one column to the data indicating whether a fire occurred for each observation ('TRUE' for area>0 and 'FALSE' for area==0).
Forest$fire = factor(Forest$area>0)

Forest = Forest %>% mutate(fire=area>0)
str(Forest)

Forest = Forest %>% mutate(fire=area>0)
Forest$fire = factor(Forest$fire)
str(Forest)


# Task 5

# What is the mean area/wind/temp/RH per month?
tapply(Forest$area, Forest$month, mean) # group by month values and apply mean on area column of each group
tapply(Forest$wind, Forest$month, mean)
tapply(Forest$temp, Forest$month, mean)
tapply(Forest$RH, Forest$month, mean)

Forest %>% group_by(month) %>% summarise(mean_area=mean(area), mean_wind=mean(wind), mean_temp=mean(temp), mean_rh=mean(RH))

# How many observations are there in each month? 
table(Forest$month)

Forest %>% group_by(month) %>% summarise(count=n())

# How many observations are there with a fire in each month?
table(Forest[Forest$area>0, ]$month)

Forest %>% filter(area>0) %>% group_by(month) %>% summarise(count=n())

# What is the probability of a fire in each month? 
table(Forest[Forest$area>0, ]$month) / table(Forest$month)

Forest %>% group_by(month) %>% summarise(prob=sum(area>0)/n())
