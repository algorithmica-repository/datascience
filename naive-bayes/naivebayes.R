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
# to get nice looking tables
library(pander)

# Set seed for reproducibility
set.seed(1234)

frqtab = function(x, caption) {
  round(100*prop.table(table(x)), 1)
}

# Read the data
setwd("E:/data analytics/datasets")
sms_raw = read.table(unz("smsspamcollection.zip","SMSSpamCollection"),
                      header=FALSE, sep="\t", quote="", stringsAsFactors=FALSE)
sms_raw = sms_raw[sample(nrow(sms_raw)),]

# Explore the dataset
dim(sms_raw)
str(sms_raw)
head(sms_raw)

colnames(sms_raw) = c("type", "text")
sms_raw$type = factor(sms_raw$type)

# Preparing the dataset
sms_corpus = Corpus(VectorSource(sms_raw$text))

inspect(sms_corpus[1:10])

#To avoid the issue with DocumentTermMatrix method, use one of following solutions:
#1) Adding content_transformer avoids the type conversion issue with non-standard transformations
#2) Add the tm_map(PlainTextDocument) after all the cleaning is done

#getTransformations() returns standard transformations

sms_corpus_clean = sms_corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace) %>%
  tm_map(stemDocument) 

inspect(sms_corpus_clean[1:10])

#Generating the training and test partitions from raw_data
#About createDataPartition:
#y = what output we want to split on, which is this case are the two types of messages (SPAM and non Spam). 
#p specifies the proportion of data that will exist in each chunk after splitting the data, 
#in this case we split into two chunks of 75% and 25%. 
#We then subset the data using the output from the createDataPartition function.

train_index = createDataPartition(sms_raw$type, p=0.75, list=FALSE)
sms_raw_train = sms_raw[train_index,]
sms_raw_test = sms_raw[-train_index,]

#Explore the training and test datasets
dim(sms_raw_train)
dim(sms_raw_test)

ft_orig = frqtab(sms_raw$type)
ft_train = frqtab(sms_raw_train$type)
ft_test = frqtab(sms_raw_test$type)
ft_df = as.data.frame(cbind(ft_orig, ft_train, ft_test))
colnames(ft_df) <- c("Original", "Training set", "Test set")
pander(ft_df, style="rmarkdown",
       caption=paste0("Comparison of SMS type frequencies among datasets"))

sms_corpus_clean_train = sms_corpus_clean[train_index]
sms_corpus_clean_test = sms_corpus_clean[-train_index]

#Feature Reduction: 
#Remove the features whose length is <=2 and 
#Remove the features that appear sparse in 98% of documents
sms_train = DocumentTermMatrix(sms_corpus_clean_train,control=list(minWordLength=2))
sms_train = removeSparseTerms(sms_train,0.98)
sms_test = DocumentTermMatrix(sms_corpus_clean_test,control=list(minWordLength=2))
sms_test = removeSparseTerms(sms_test,0.98)
dim(sms_train)
dim(sms_test)
inspect(sms_train[1:10,1:10])
inspect(sms_test[1:10,1:10])

# Convert the dtm into boolean values instead of term frequencies
convert_counts <- function(x) {
  x = ifelse(x > 0, 1, 0)
  x = factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}
sms_train = sms_train %>% apply(MARGIN=2, FUN=convert_counts)
sms_test = sms_test %>% apply(MARGIN=2, FUN=convert_counts)
dim(sms_train)
dim(sms_test)
sms_train[1:10,1:10]
sms_test[1:10,1:10]


#Train the model
sms_model = naiveBayes(sms_train, sms_raw_train$type)
sms_model

#Test the model
sms_predict = predict(sms_model, sms_test)

cm = confusionMatrix(sms_predict, sms_raw_test$type, positive="spam")
cm



