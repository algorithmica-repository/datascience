library(arules)
library(ggplot2)
library(arulesViz)
library(dplyr)

setwd("E:/apriori/1000")
transactions = read.csv("1000i.csv", header = F, stringsAsFactors=FALSE)
dim(transactions)
str(transactions)
names(transactions) = c("id","qty","item")

transactions1 = transactions %>% select(id,item)
dim(transactions1)


trans = as(split(transactions1$item, transactions1$id), "transactions")
trans
inspect(trans[1:5])

rules = apriori(trans, parameter = list(supp=0.01,conf=0.6))
inspect(rules)

rules1 = apriori(trans, parameter = list(supp=0.01,conf=0.6),
                 appearance = list(default="lhs",rhs="24"),
                 control = list(verbose=F))
inspect(rules1)

rules=apriori(trans, parameter=list(supp=0.001,conf = 0.08), 
               appearance = list(default="lhs",rhs="coldplay"),
               control = list(verbose=F))


rules=apriori(trans, parameter=list(supp=0.001,conf = 0.15,minlen=2), 
               appearance = list(default="rhs",lhs="beck"),
               control = list(verbose=F))
rules=sort(rules, decreasing=TRUE,by="confidence")
inspect(rules[1:5])
