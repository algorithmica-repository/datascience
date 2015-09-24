library(dplyr)
library(SnowballC)
library(caret)
library(tm)
library(RWeka)
library(stringi)
library(ggplot2)

//Set the path of jre to use RWeka library
Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre7\\')

setwd("E:/data analytics/datasets/emails")

files=list.files(pattern="*.*", recursive = TRUE)
emails = character(length(files))

for(i in 1:length(files)) {
  emails[i] = stri_flatten(readLines(paste(getwd(), files[i],sep='/')))
}

corpus = Corpus(VectorSource(emails))
#some versions of tm, inspect is not working
inspect(corpus)
#try this insted of inspect
corpus[[2]]

custom_transform =  content_transformer(function(x, from, to) gsub(from, to, x))

corpus_clean =  tm_map(corpus,custom_transform,"/"," ")

corpus_clean =  tm_map(corpus_clean, content_transformer(tolower))

corpus_clean =  tm_map(corpus_clean, removeNumbers)

corpus_clean =  tm_map(corpus_clean, removePunctuation)

corpus_clean =  tm_map(corpus_clean, removeWords, stopwords("english"))

corpus_clean =  tm_map(corpus_clean, stripWhitespace)

corpus_clean =  tm_map(corpus_clean, stemDocument)

inspect(corpus_clean)
corpus_clean[[2]]

//doing all transformations with less painful code
corpus_clean = corpus %>%
  tm_map(custom_transform,"/"," ") %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(stripWhitespace) %>%
  tm_map(stemDocument) 


dtm_unigrams = DocumentTermMatrix(corpus_clean,control=list(weighting=weightTf))
inspect(dtm_unigrams[1,1:5])

ngrams = function(x,n=1) NGramTokenizer(x, Weka_control(min = n, max = n))
dtm_bigrams = DocumentTermMatrix(corpus_clean,control=list(weighting=weightTf, tokenize= function(x) ngrams(x,2)))
inspect(dtm_bigrams[1,1:5])

write.csv(as.matrix(dtm_unigrams), "dtm_unigrams.csv")
write.csv(as.matrix(dtm_bigrams), "dtm_bigrams.csv")