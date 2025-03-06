# ========================================================
# Natural Language Processing with Disaster Tweets - v7
# Model: TF-IDF + XGBoost
# ========================================================

## 1. Load Required Libraries ----
library(caret)        # Machine learning utilities
library(caTools)      # Data splitting
library(ggplot2)      # Data visualization
library(stringr)      # String operations
library(textstem)     # Lemmatization
library(tm)           # Text processing (Corpus, DTM)
library(tokenizers)   # Tokenization
library(tidyverse)    # Data manipulation
library(xgboost)      # XGBoost model

## 2. Load Datasets ----
train_df <- read.csv("train.csv", stringsAsFactors = FALSE)
test_df <- read.csv("test.csv", stringsAsFactors = FALSE)

## 3. Exploratory Data Analysis ----
# Summary of dataset
print(head(train_df))
print(summary(train_df))

# Class distribution
ggplot(train_df, aes(x = factor(target))) +
  geom_bar(fill = c("red", "blue")) +
  labs(title = "Distribution of Disaster vs. Non-Disaster Tweets",
       x = "Tweet Type", y = "Count") +
  scale_x_discrete(labels = c("Non-Disaster", "Disaster"))

## 4. Text Preprocessing ----
clean_text <- function(text) {
  text <- tolower(text)
  text <- gsub("http[^[:space:]]*", "", text)  # Remove URLs
  text <- gsub("[^[:alpha:][:space:]]*", "", text)  # Remove punctuation & numbers
  words <- unlist(strsplit(text, " "))
  words <- words[!(words %in% stopwords("en"))]  # Remove stopwords
  return(paste(words, collapse = " "))
}

# Apply cleaning function to both datasets
train_df$text <- sapply(train_df$text, clean_text)
test_df$text <- sapply(test_df$text, clean_text)

## 5. Tokenization & Lemmatization ----
tokenized_text <- strsplit(train_df$text, " ")
lemmatized_text <- lapply(tokenized_text, lemmatize_strings)
train_df$text <- sapply(lemmatized_text, paste, collapse = " ")

tokenized_test_text <- strsplit(test_df$text, " ")
lemmatized_test_text <- lapply(tokenized_test_text, lemmatize_strings)
test_df$text <- sapply(lemmatized_test_text, paste, collapse = " ")

## 6. Feature Engineering (TF-IDF) ----
corpus <- Corpus(VectorSource(train_df$text))
dtm <- DocumentTermMatrix(corpus)
matrix_dtm <- as.matrix(dtm)

# Compute TF-IDF
idf <- log2(nrow(matrix_dtm) / rowSums(matrix_dtm > 0))
tfidf_matrix <- sweep(matrix_dtm, 2, idf, '*')

## 7. Model Training (XGBoost) ----
set.seed(123)

# Train-test split
split_train_val <- sample.split(train_df$target, SplitRatio = 0.9)
train_val_data <- tfidf_matrix[split_train_val, ]
train_val_target <- train_df$target[split_train_val]

split_train <- sample.split(train_val_target, SplitRatio = 0.8)
train_data <- train_val_data[split_train, ]
val_data <- train_val_data[!split_train, ]
train_target <- train_val_target[split_train]
val_target <- train_val_target[!split_train]

test_data <- tfidf_matrix[!split_train_val, ]
test_target <- train_df$target[!split_train_val]

# Convert to DMatrix format
dtrain <- xgb.DMatrix(data = as.matrix(train_data), label = train_target)
dval <- xgb.DMatrix(data = as.matrix(val_data), label = val_target)
dtest <- xgb.DMatrix(data = as.matrix(test_data), label = test_target)

# Handle class imbalance
imbalance_ratio <- sum(train_target == 0) / sum(train_target == 1)

# XGBoost parameters
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.3,
  max_depth = 6,
  scale_pos_weight = imbalance_ratio
)

# Train XGBoost model
num_rounds <- 100
watchlist <- list(train = dtrain, val = dval)
model <- xgb.train(params = params, data = dtrain, nrounds = num_rounds, 
                   watchlist = watchlist, early_stopping_rounds = 10)

# Feature importance
importance_matrix <- xgb.importance(model = model)
print(importance_matrix)

# Predictions on test data
preds <- predict(model, dtest)
preds_binary <- ifelse(preds > 0.5, 1, 0)

# Model evaluation
cm <- confusionMatrix(as.factor(preds_binary), as.factor(test_target))
print(cm)

# Extract and print F1 Score
f1_score <- cm$byClass["F1"]
print(paste("F1 Score:", f1_score))

## 8. Prepare Submission ----
# Preprocess test data for TF-IDF transformation
corpus_test <- Corpus(VectorSource(test_df$text))
dtm_test <- DocumentTermMatrix(corpus_test, control = list(dictionary = Terms(dtm)))

# Combine train and test DTM to ensure feature alignment (Fix for feature mismatch)
combined_dtm <- rbind(dtm, dtm_test)
matrix_dtm_test <- as.matrix(combined_dtm[(nrow(dtm) + 1):nrow(combined_dtm), ])

# Compute TF-IDF using the training IDF
tfidf_matrix_test <- sweep(matrix_dtm_test, 2, idf, '*')
dtest_new <- xgb.DMatrix(data = as.matrix(tfidf_matrix_test))

# Make predictions
preds_new <- predict(model, dtest_new)
preds_binary_new <- ifelse(preds_new > 0.5, 1, 0)

# Create submission file
submission <- data.frame(id = test_df$id, target = preds_binary_new)
write.csv(submission, "submission4.csv", row.names = FALSE)