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
library(wordcloud)
library(RWeka)

setwd("D:\\social-comments")

trainSet = read.csv("train.csv", header = TRUE, na.strings=c("NA",""), stringsAsFactors = F)
dim(trainSet)
str(trainSet)
head(trainSet)

comment_corpus = Corpus(VectorSource(trainSet$Comment))
as.character(comment_corpus[[11]])
for(i in 1:length(comment_corpus))
  print(as.character(comment_corpus[[i]]))

# comment_corpus = tm_map(comment_corpus, content_transformer(tolower)) 
# comment_corpus = tm_map(comment_corpus, removeNumbers)
# comment_corpus = tm_map(comment_corpus, removePunctuation)
# comment_corpus = tm_map(comment_corpus, removeWords, stopwords(kind="en"))
# comment_corpus = tm_map(comment_corpus, stripWhitespace)
# comment_corpus = tm_map(comment_corpus, stemDocument)

comment_corpus_clean = comment_corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace) %>%
  tm_map(stemDocument) 

for(i in 1:length(comment_corpus_clean))
  print(as.character(comment_corpus_clean[[i]]))

comment_train1 = DocumentTermMatrix(comment_corpus_clean)
class(comment_train1)
write.csv(as.matrix(comment_train1), "dtm_unigrams.csv")

ngrams = function(x,n=1) NGramTokenizer(x, Weka_control(min = n, max = n))
#ngrams("abc def xy pqr", 3)

comment_train2 = DocumentTermMatrix(comment_corpus_clean, 
                                    control=list(tokenize= function(x) ngrams(x,2)))
class(comment_train1)
write.csv(as.matrix(comment_train2), "dtm_bigrams.csv")
inspect(comment_train1[,1:10])

freq = sort(colSums(as.matrix(comment_train1)), decreasing = T)
X11()
wordcloud( names(freq), freq)
# 
# df = data.frame(words = names(freq), freq=freq)
# wordcloud(df$words,df$freq)


findFreqTerms(comment_train1,lowfreq = 2)
X11()
ggplot(df, aes(words, freq)) + geom_bar(stat="identity")
ggplot(subset(df, freq>1), aes(words, freq)) + geom_bar(stat="identity")

findAssocs(comment_train1, "contrast", corlimit=0.90)

comment_train2 = removeSparseTerms(comment_train1, 0.99)
dim(dtmss)
write.csv(as.matrix(dtmss), "dtm_unigrams1.csv")

library(fpc)   
library(cluster)  
dtms <- removeSparseTerms(dtm, 0.15)    
d <- dist(t(comment_train1), method="euclidian")   
kfit <- kmeans(d, 2)   
clusplot(as.matrix(d), kfit$cluster, color=T, shade=T, labels=2, lines=0)   

comment_train1 = DocumentTermMatrix(comment_corpus_clean,control = list(weighting=))
dim(comment_train1)
inspect(comment_train1[,1:10])

comment_train = removeSparseTerms(comment_train,0.98)

# Convert the dtm into boolean values instead of term frequencies
convert_counts <- function(x) {
  x = ifelse(x > 0, 1, 0)
  x = factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}

comment_train = comment_train %>% apply(MARGIN=2, FUN=convert_counts)
dim(comment_train)
comment_train[1:10,1:10]

trainSet$Insult = as.factor(trainSet$Insult)
#Train the model
ctrl = trainControl(method = "cv",  number = 10)
set.seed(1234)

nb_model = train(as.matrix(comment_train2), trainSet$Insult,
                    method = "knn", trControl = ctrl)

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
