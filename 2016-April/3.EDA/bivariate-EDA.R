library(ggplot2)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")
titanic = read.csv("train.csv",na.strings=c("NA",""))
dim(titanic)
str(titanic)

# converting the types of features
titanic$Survived = as.factor(titanic$Survived)
titanic$Pclass = as.factor(titanic$Pclass)
titanic$Family = titanic$SibSp + titanic$Parch + 1

# Comparing Survived and passenger class using chisquare test, contigency table and bar charts
xtabs(~Survived + Pclass, data=titanic)
ggplot(titanic, aes(x = Survived, fill = Pclass)) + geom_bar()
ggplot(titanic, aes(x = Survived, fill = Pclass)) + geom_bar(position = "fill")
chisq.test(titanic$Survived, titanic$Pclass)

# Comparing Survived and Sex using  using chisquare test, contigency table and histograms
xtabs(~Survived + Sex, data=titanic)
ggplot(titanic, aes(x = Survived, fill = Sex)) + geom_bar()
ggplot(titanic, aes(x = Survived, fill = Sex)) + geom_bar(position="fill")
ggplot(titanic, aes(x = Sex, fill = Survived)) + geom_bar(position="fill")
chisq.test(titanic$Survived, titanic$Sex)

# Comparing Survived and Embarked using table and bar charts
xtabs(~Survived + Embarked, data=titanic)
ggplot(titanic, aes(x = Survived, fill = Embarked)) + geom_bar(position = "fill")
chisq.test(titanic$Survived,titanic$Embarked)

# Comparing Age and Survived using boxplot, histogram and density plot 
ggplot(titanic, aes(x = Survived, y = Age)) + geom_boxplot()
ggplot(titanic, aes(x = Age)) + geom_histogram() + facet_grid(Survived ~ .)
ggplot(titanic, aes(x = Age, color = Survived)) + geom_density() 


# Comparing Survived and Fare using boxplots 
ggplot(titanic, aes(x = Survived, y = Fare)) + geom_boxplot() 
ggplot(titanic, aes(x = Fare)) + geom_histogram() + facet_grid(Survived ~ .)
ggplot(titanic, aes(x = Fare, color = Survived)) + geom_density() 

# Comparing Survived and Family using boxplots
ggplot(titanic, aes(x = Survived, y = Family)) + geom_boxplot()
ggplot(titanic, aes(x = Family)) + geom_histogram() + facet_grid(Survived ~ .)
ggplot(titanic, aes(x = Family, color = Survived)) + geom_density()

# Comparing Sibsp and Fare using scatterplot
cov(titanic$SibSp, titanic$Fare)
cor(titanic$SibSp, titanic$Fare)
ggplot(titanic, aes(x = SibSp, y = Fare)) + geom_point()
