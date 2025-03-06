# Natural Language Processing with Disaster Tweets_v7

## Preliminary Tasks

#Load libraries
library(caret)
library(caTools)
library(e1071)
library(ggplot2)
library(Matrix)
library(randomForest)
library(ROSE)
library(SnowballC)
library(stringr)
library(textclean)
library(textstem)
library(textTinyR)
library(tidytext)
library(tidyverse)
library(tm)
library(tokenizers)
library(wordcloud)
library(xgboost)

#Load the dataset
train_df <- read.csv(
  "train.csv", 
  stringsAsFactors = FALSE
)

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

#Inspect Cleaned Data
head(train_df$text)

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

## Build the Model: Gradient Boosting

#Partition Data (training/validation/testing)
#90% for training + validation
set.seed(123)
split_train_val <- sample.split(train_df$target, SplitRatio = 0.9)  # 90% for training + validation
train_val_data <- tfidf_matrix[split_train_val, ]
train_val_target <- train_df$target[split_train_val]

#80% of 90% for training
split_train <- sample.split(train_val_target, SplitRatio = 0.8)
train_data <- train_val_data[split_train, ]
val_data <- train_val_data[!split_train, ]
train_target <- train_val_target[split_train]
val_target <- train_val_target[!split_train]

test_data <- tfidf_matrix[!split_train_val, ]
test_target <- train_df$target[!split_train_val]

# Convert to 'DMatrix' format
dtrain <- xgb.DMatrix(data = as.matrix(train_data), label = train_target)
dval <- xgb.DMatrix(data = as.matrix(val_data), label = val_target)
dtest <- xgb.DMatrix(data = as.matrix(test_data), label = test_target)

#Hande class imbalance
imbalance_ratio <- sum(train_target == 0) / sum(train_target == 1)

# Set parameters for early stopping and class imbalance handling
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.3,
  max_depth = 6,
  scale_pos_weight = imbalance_ratio
)

# Train the model
num_rounds <- 100
watchlist <- list(train = dtrain, val = dval)
model <- xgb.train(params = params, data = dtrain, nrounds = num_rounds, watchlist = watchlist, early_stopping_rounds = 10)

# Extract feature importance
importance_matrix <- xgb.importance(model = model)
print(importance_matrix)

#Predict on validation data
preds <- predict(model, dtest)
preds_binary <- ifelse(preds > 0.5, 1, 0)

#Evaluate the model
confusionMatrix(as.factor(preds_binary), as.factor(test_target))

## Create Submission

#Import test dataset
test_df <- read.csv("test.csv", stringsAsFactors = FALSE)

### Duplicate Text Preprocessing for Competition Data

#Text preprocessing
test_df$text <- tolower(test_df$text)
test_df$text <- gsub("http[^[:space:]]*", "", test_df$text)
test_df$text <- gsub("[^[:alpha:][:space:]]*", "", test_df$text)
test_df$text <- sapply(test_df$text, remove_stopwords)

#Tokenize and lemmatize
tokenized_test_text <- strsplit(test_df$text, " ")
lemmatized_test_text <- lapply(tokenized_test_text, lemmatize_strings)
lemmatized_test_text_cleaned <- lapply(lemmatized_test_text, function(tweet) tweet[tweet != ""])

# Convert into a character vector
tweets_test_vector <- sapply(lemmatized_test_text_cleaned, paste, collapse = " ")

# Create corpus
corpus_test <- Corpus(VectorSource(tweets_test_vector))

#Create DTM
dtm_test <- DocumentTermMatrix(corpus_test, control = list(dictionary = Terms(dtm)))

#Confirm data structure duplication
combined_dtm <- rbind(dtm, dtm_test)
matrix_dtm_test <- as.matrix(combined_dtm[(nrow(dtm) + 1):nrow(combined_dtm), ])

#Compute TF-IDF
tfidf_matrix_test <- sweep(matrix_dtm_test, 2, idf, '*')

#Convert to DMatrix format
dtest_new <- xgb.DMatrix(data = as.matrix(tfidf_matrix_test))

#Predict on competition data
preds_new <- predict(model, dtest_new)
preds_binary_new <- ifelse(preds_new > 0.5, 1, 0)

# Evaluate the model using confusion matrix
cm <- confusionMatrix(as.factor(preds_binary), as.factor(test_target))

# Extract and print the F1 score
f1_score <- cm$byClass["F1"]
print(paste("F1 Score:", f1_score))

#Create submission dataframe
submission <- data.frame(id = test_df$id, target = preds_binary_new)

# Write submission to CSV
write.csv(submission, "submission4.csv", row.names = FALSE)