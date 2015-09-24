library(dplyr)
library(SnowballC)
library(caret)
library(tm)
library(RWeka)
library(stringi)
library(wordcloud)
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

dtm_unigrams_reduced = removeSparseTerms(dtm_unigrams,0.98)

findFreqTerms(dtm_unigrams, lowfreq=10)

findAssocs(dtm_unigrams, "free",.5) 

freq = sort(colSums(as.matrix(dtm_unigrams)), decreasing=TRUE)
head(freq)
wf = data.frame(word=names(freq), freq=freq)
head(wf)
ggplot(wf,aes(word, freq)) + geom_bar(stat="identity")

wordcloud(names(freq),freq,min.freq=40)
wordcloud(names(freq),freq,max.words=100)
wordcloud(names(freq), freq, min.freq=100, colors=brewer.pal(6, "Dark2"))
#By default the most frequent words have a scale of 4 and the least have a scale of 0.5
wordcloud(names(freq), freq, min.freq=100, scale=c(5, .1), colors=brewer.pal(6, "Dark2"))
#We can change the proportion of words that are rotated by 90 degrees from the default 10% to,
#say, 20% using rot.per=0.2.
wordcloud(names(freq), freq, min.freq=100, rot.per=0.2, colors=brewer.pal(6, "Dark2"))

pca = preProcess(as.matrix(dtm_unigrams), method=c("pca"))
pca
pca$rotation
dtm_unigrams_reduced = predict(pca,as.matrix(dtm_unigrams))
dim(dtm_unigrams_reduced)


