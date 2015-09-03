library(tm)
library(dplyr)
library(SnowballC)
library(wordcloud)
path = file.path("E:/data analytics/datasets/20news-bydate-test/comp.graphics")
corpus=Corpus(DirSource(path))
typeof(corpus)
inspect(corpus)
inspect(corpus[1:5])
as.character(corpus[[1]])

corpus_clean = corpus %>% tm_map(removeNumbers) %>%
                tm_map(removePunctuation)%>%
                tm_map(content_transformer(tolower)) %>%
                tm_map(stripWhitespace) %>%
                tm_map(removeWords, stopwords("en")) %>%
                tm_map(stemDocument)

inspect(corpus_clean)

dtm = DocumentTermMatrix(corpus_clean)

class(dtm)

inspect(dtm[1:25,25:30])

dtm = removeSparseTerms(dtm,0.9)

dim(dtm)
inspect(dtm)
inspect(dtm[1:25,1:10])

findFreqTerms(dtm, 100)
findAssocs(dtm, "sweep",0.9)

freq=colSums(as.matrix(dtm))
length(freq)
sort(freq,decreasing = TRUE)

set.seed(120)
wordcloud(names(freq),freq,max.words = 50)
wordcloud(names(freq),freq,min.freq = 100)
