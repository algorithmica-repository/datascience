# to simplify selections
library(dplyr)
# for stemming the words
library(SnowballC)
# libraries required by caret
library(klaR)
library(e1071)
# for the Naive Bayes modelling
library(caret)
# to process the text into a corpus
library(tm)
# for working with caret
library(caret)


setwd("E:/data analytics/kaggle/insult-detection/data")

trainSet = read.csv("train.csv", header = TRUE, na.strings=c("NA",""))
dim(trainSet)
str(trainSet)
head(trainSet)

comment_corpus = Corpus(VectorSource(trainSet$Comment))
inspect(comment_corpus)
comment_corpus_clean = comment_corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace) %>%
  tm_map(stemDocument) 

inspect(comment_corpus_clean)

trainSet$Insult = factor(trainSet$Insult)

#Feature Reduction: 
#Remove the features whose length is <=2 and 
#Remove the features that appear sparse in 98% of documents
comment_train = DocumentTermMatrix(comment_corpus_clean)
comment_train = removeSparseTerms(comment_train,0.98)
dim(comment_train)
inspect(comment_train[1:10,1:10])


# Convert the dtm into boolean values instead of term frequencies
convert_counts <- function(x) {
  x = ifelse(x > 0, 1, 0)
  x = factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}

comment_train = comment_train %>% apply(MARGIN=2, FUN=convert_counts)
dim(comment_train)
comment_train[1:10,1:10]


#Train the model
ctrl = trainControl(method = "cv",  number = 10)
set.seed(1234)

naive_model = train(comment_train, trainSet$Insult,
                    method = "nb", trControl = ctrl)

naive_model



testSet = read.table("test_with_solutions.csv", sep = ",", header = TRUE)
dim(testSet)
str(testSet)
head(testSet)
summary(testSet)

comment_corpus = Corpus(VectorSource(testSet$Comment))
inspect(comment_corpus)
comment_corpus_clean = comment_corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace) %>%
  tm_map(stemDocument) 

inspect(comment_corpus_clean)

#Feature Reduction: 
#Remove the features whose length is <=2 and 
#Remove the features that appear sparse in 98% of documents
comment_test = DocumentTermMatrix(comment_corpus_clean)
comment_test = removeSparseTerms(comment_test,0.90)
dim(comment_test)
inspect(comment_test[1:10,1:10])


# Convert the dtm into boolean values instead of term frequencies
convert_counts <- function(x) {
  x = ifelse(x > 0, 1, 0)
  x = factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}

comment_test = comment_test %>% apply(MARGIN=2, FUN=convert_counts)
dim(comment_test)
comment_test[1:10,1:10]


predict1 = predict(naive_model, newdata = comment_test)
confusionMatrix(predict1,testSet$Insult, positive="1")
