# Natural Language Processing with Disaster Tweets_v3.2

## Preliminary Tasks

#Load libraries
library(caret)
library(caTools)
#library(e1071)
library(ggplot2)
#library(Matrix)
#library(randomForest)
#library(ROSE)
#library(SnowballC)
library(stringr)
#library(textclean)
library(textstem)
#library(textTinyR)
library(tidytext)
#library(tidyverse)
library(tm)
#library(tokenizers)
#library(wordcloud)
library(xgboost)

#Load the datasets
train_df <- read.csv("train.csv", stringsAsFactors = FALSE)
test_df <- read.csv("test.csv", stringsAsFactors = FALSE)

## Exploratory Data Analysis

#Summarize
head(train_df)
summary(train_df)

#Analyze and visualize distribution
table(train_df$target)
ggplot(train_df, aes(x = factor(target))) +
  geom_bar(fill = c("red", "blue")) +
  labs(title = "Distribution of Disaster Tweets vs. Non-Disaster Tweets",
       x = "Tweet Type",
       y = "Count") +
  scale_x_discrete(labels = c("Non-Disaster", "Disaster"))

## Text Preprocessing

### Custom Cleaning

#### Create 'americanize'|'anglicize' Transformers

#Load dictionary data
spellings <- read.csv("uk-us-spelling-list.csv", stringsAsFactors = FALSE)

#Create dictionaries
us_to_uk <- setNames(spellings$UK, spellings$US)
uk_to_us <- setNames(spellings$US, spellings$UK)

#Define 'americanize' function
americanize <- function(text) {
  words <- strsplit(text, "\\s")[[1]]
  words <- ifelse(words %in% names(uk_to_us), uk_to_us[words], words)
  paste(words, collapse = " ")
}

#Define 'anglicize' function
anglicize <- function(text) {
  words <- strsplit(text, "\\s")[[1]]
  words <- ifelse(words %in% names(us_to_uk), us_to_uk[words], words)
  paste(words, collapse = " ")
}

#"Americanize" Text
train_df$text <- sapply(train_df$text, americanize)

### Text Cleaning

#Convert to lowercase
train_df$text <- tolower(train_df$text)

#Remove URLs
train_df$text <- gsub("http[^[:space:]]*", "", train_df$text)

#Remove numbers and punctuation
train_df$text <- gsub("[^[:alpha:][:space:]]*", "", train_df$text)

#Remove stopwords
stopwords_list <- stopwords("en")
remove_stopwords <- function(text) {
  words <- unlist(strsplit(text, " "))
  words <- words[!(words %in% stopwords_list)]
  return(paste(words, collapse = " "))
}
train_df$text <- sapply(train_df$text, remove_stopwords)

#Strip whitespace
train_df$text <- gsub("\\s+", " ", train_df$text)

#Inspect cleaned data
head(train_df$text)

### Duplicate Text Cleaning for Test Data
test_df$text <- sapply(test_df$text, americanize)
test_df$text <- tolower(test_df$text)
test_df$text <- gsub("http[^[:space:]]*", "", test_df$text)
test_df$text <- gsub("[^[:alpha:][:space:]]*", "", test_df$text)
test_df$text <- sapply(test_df$text, remove_stopwords)
test_df$text <- gsub("\\s+", " ", test_df$text)
head(test_df$text)

### Tokenize and Lemmatize Text

#Tokenize and inspect text
tokenized_text <- strsplit(train_df$text, " ")
head(tokenized_text)

#Lemmatize and inspect
lemmatized_text <- lapply(tokenized_text, lemmatize_strings)
head(lemmatized_text)

#Remove empty strings
lemmatized_text_cleaned <- lapply(lemmatized_text, function(tweet) tweet[tweet != ""])

#Inspect lemmatized text, round 2
head(lemmatized_text_cleaned)

### Duplicate Tokenization and Lemmatization for Test Data

tokenized_text_test <- strsplit(test_df$text, " ")
lemmatized_text_test <- lapply(tokenized_text_test, lemmatize_strings)
lemmatized_text_cleaned_test <- lapply(lemmatized_text_test, function(tweet_test) tweet_test[tweet_test != ""])
head(lemmatized_text_cleaned_test)

## Feature Engineering

#Convert text to a character vector
tweets_vector <- sapply(lemmatized_text_cleaned, paste, collapse = " ")

#Create corpus
corpus <- Corpus(VectorSource(tweets_vector))

# Create a DTM
dtm <- DocumentTermMatrix(corpus)

# Convert the DTM to a matrix
matrix_dtm <- as.matrix(dtm)

# Compute Inverse Document Frequency (IDF)
idf <- log2(nrow(matrix_dtm) / rowSums(matrix_dtm > 0))

# Compute and Summarize TF-IDF matrix
tfidf_matrix <- sweep(matrix_dtm, 2, idf, '*')
list(
  dimensions = dim(tfidf_matrix),
  num_nonzero_entries = sum(tfidf_matrix > 0),
  sparsity = 1 - (sum(tfidf_matrix > 0) / (nrow(tfidf_matrix) * ncol(tfidf_matrix)))
)

### Duplicate Feature Engineering for Test Set

tweets_test_vector <- sapply(lemmatized_text_cleaned_test, paste, collapse = " ")
corpus_test <- Corpus(VectorSource(tweets_test_vector))
dtm_test <- DocumentTermMatrix(corpus_test, control = list(dictionary = Terms(dtm)))
combined_dtm <- rbind(dtm, dtm_test)
matrix_dtm_test <- as.matrix(combined_dtm[(nrow(dtm) + 1):nrow(combined_dtm), ])
tfidf_matrix_test <- sweep(matrix_dtm_test, 2, idf, '*')
dtest_new <- xgb.DMatrix(data = as.matrix(tfidf_matrix_test))

## Build the Model: Gradient Boosting

# Setting seed for reproducibility
set.seed(123)

# Partition Data
split <- sample.split(train_df$target, SplitRatio = 0.8)
train_data <- tfidf_matrix[split, ]
valid_data <- tfidf_matrix[!split, ]
train_target <- train_df$target[split]
valid_target <- train_df$target[!split]

# Convert data to DMatrix format
dtrain <- xgb.DMatrix(data = as.matrix(train_data), label = train_target)
dvalid <- xgb.DMatrix(data = as.matrix(valid_data), label = valid_target)

# Set parameters
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.3,
  max_depth = 6
)

# Train the model
num_rounds <- 100
watchlist <- list(train = dtrain, valid = dvalid)
model <- xgb.train(params = params, data = dtrain, nrounds = num_rounds, watchlist = watchlist)

# Predict on validation data
preds <- predict(model, dvalid)

# Convert predictions to binary format
preds_binary <- ifelse(preds > 0.5, 1, 0)

# Evaluate the model using confusion matrix
cm <- confusionMatrix(as.factor(preds_binary), as.factor(valid_target))
print(cm)

# Extract F1 score
f1_score <- cm$byClass["F1"]
print(f1_score)

## Create Submission

# Predict on competition data
preds_new <- predict(model, dtest_new)
preds_binary_new <- ifelse(preds_new > 0.5, 1, 0)

# Create submission dataframe
submission <- data.frame(id = test_df$id, target = preds_binary_new)

# Write submission to CSV
write.csv(submission, "submission2.csv", row.names = FALSE)