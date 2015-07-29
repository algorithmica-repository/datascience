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

# Set seed for reproducibility
set.seed(1234)

# Read the data
setwd("E:/data analytics/datasets")
sms_raw = read.table(unz("smsspamcollection.zip","SMSSpamCollection"),
                      header=FALSE, sep="\t", quote="", stringsAsFactors=FALSE)
# Explore the dataset
dim(sms_raw)
str(sms_raw)
head(sms_raw)

colnames(sms_raw) = c("type", "text")
sms_raw$type = factor(sms_raw$type)

# Preparing the dataset
sms_corpus = Corpus(VectorSource(sms_raw$text))

as.character(sms_corpus[[1]])
inspect(sms_corpus[1:10])

#To avoid the issue with DocumentTermMatrix method, use one of following solutions:
#1) Adding content_transformer avoids the type conversion issue with non-standard transformations
#2) Add the tm_map(PlainTextDocument) after all the cleaning is done

sms_corpus_clean = sms_corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace) %>%
  tm_map(stemDocument) 

inspect(sms_corpus_clean[1:10])

sms_corpus_clean = DocumentTermMatrix(sms_corpus_clean,control=list(minWordLength=2))
dim(sms_corpus_clean)
sms_corpus_clean = removeSparseTerms(sms_corpus_clean,0.98)
dim(sms_corpus_clean)
inspect(sms_corpus_clean[1:10,1:10])

# Convert the dtm into boolean values instead of term frequencies
convert_counts <- function(x) {
  x = ifelse(x > 0, 1, 0)
  x = factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}
sms_corpus_clean_binary= sms_corpus_clean %>% apply(MARGIN=2, FUN=convert_counts)
dim(sms_corpus_clean_binary)
sms_corpus_clean_binary[1:10,1:10]


#Train the model
ctrl = trainControl(method="cv", 10)
sms_model = train(sms_corpus_clean_binary, sms_raw$type, method="nb", trControl=ctrl)
sms_model


ctrl = trainControl(method="cv", 10)

sms_model = train(sms_corpus_clean_binary, sms_raw$type, method="nb", trControl=ctrl)
str(sms_model)
sms_model$trainingData
sms_model$resample
sms_model$time

#Test the model
sms_predict = predict(sms_model, sms_corpus_clean_binary, type="prob")
str(sms_predict)
head(sms_predict)

cm = confusionMatrix(sms_predict, sms_raw$type, positive="spam")
cm



