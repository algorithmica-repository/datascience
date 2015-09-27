library(arules)
library(ggplot2)
library(arulesViz)
library(dplyr)

setwd("E:/data analytics/datasets/")
lastfm = read.csv("lastfm.csv", header = TRUE, stringsAsFactors=FALSE)
dim(lastfm)
str(lastfm)

ds = lastfm %>% select(user, artist) %>% unique()
dim(ds)
head(ds)

trans = as(split(ds$artist, ds$user), "transactions")
trans
inspect(trans[1:5])
itemFrequency(trans, type="absolute")
itemFrequencyPlot(trans, support=0.075, type="absolute")
itemFrequency(trans)
itemFrequencyPlot(trans, support=0.075)


rules = apriori(trans, parameter=list(support=0.01, confidence=0.5))
inspect(rules)
sort(rules, by="confidence", decreasing=TRUE)

rules=apriori(trans, parameter=list(supp=0.001,conf = 0.08), 
               appearance = list(default="lhs",rhs="coldplay"),
               control = list(verbose=F))
rules=sort(rules, decreasing=TRUE,by="confidence")
inspect(rules[1:5])

rules=apriori(trans, parameter=list(supp=0.001,conf = 0.15,minlen=2), 
               appearance = list(default="rhs",lhs="beck"),
               control = list(verbose=F))
rules=sort(rules, decreasing=TRUE,by="confidence")
inspect(rules[1:5])

plot(rules,method="graph",interactive=TRUE,shading=NA)

inspect(subset(rules, subset=lift>8))

inspect(sort(subset(rules, subset=lift>8), by="confidence"))
