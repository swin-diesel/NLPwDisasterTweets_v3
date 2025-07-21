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

## 6. Feature Engineering (TF‑IDF) ----------
set.seed(123)

# Outer 80/20 split
train_idx   <- sample(seq_len(nrow(train_df)), 0.80 * nrow(train_df))
train_split <- train_df[train_idx, ]
test_split  <- train_df[-train_idx, ]

# Inner 80/20 split for early‑stopping
inner_idx   <- sample(seq_len(nrow(train_split)), 0.80 * nrow(train_split))
train_inner <- train_split[inner_idx, ]
val_inner   <- train_split[-inner_idx, ]

# DTM on train_inner only
corpus_train <- Corpus(VectorSource(train_inner$text))
dtm_train    <- DocumentTermMatrix(corpus_train)
mat_train    <- as.matrix(dtm_train)

# Correct IDF vector (document frequency is column‑wise)
idf_vec      <- log2(nrow(mat_train) / pmax(1, colSums(mat_train > 0)))
tfidf_train  <- sweep(mat_train, 2, idf_vec, "*")

# Helper to guarantee identical feature set & order
fix_columns <- function(mat, ref_names) {
  missing <- setdiff(ref_names, colnames(mat))
  if (length(missing)) {                       # add zero‑filled cols
    mat <- cbind(mat, matrix(0, nrow(mat), length(missing),
                             dimnames = list(NULL, missing)))
  }
  mat[, ref_names]                             # reorder
}

# Build val / test DTMs, then map to training vocabulary ----------
ctrl      <- list(dictionary = Terms(dtm_train))
mat_val   <- as.matrix(DocumentTermMatrix(Corpus(VectorSource(val_inner$text)),  control = ctrl))
mat_test  <- as.matrix(DocumentTermMatrix(Corpus(VectorSource(test_split$text)), control = ctrl))

# Pad & reorder so cols match training set
mat_val   <- fix_columns(mat_val,  colnames(mat_train))
mat_test  <- fix_columns(mat_test, colnames(mat_train))

# Apply same IDF weighting
tfidf_val   <- sweep(mat_val,  2, idf_vec, "*")
tfidf_test  <- sweep(mat_test, 2, idf_vec, "*")

y_train <- train_inner$target
y_val   <- val_inner$target
y_test  <- test_split$target

## 7. Model Training (XGBoost) ----
dtrain <- xgb.DMatrix(tfidf_train, label = y_train)
dval   <- xgb.DMatrix(tfidf_val,   label = y_val)
dtest  <- xgb.DMatrix(tfidf_test,  label = y_test)

imbalance_ratio <- sum(y_train == 0) / sum(y_train == 1)

params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.3,
  max_depth = 6,
  scale_pos_weight = imbalance_ratio
)

num_rounds <- 100
watchlist  <- list(train = dtrain, val = dval)

model <- xgb.train(
  params, dtrain, nrounds = num_rounds,
  watchlist = watchlist, early_stopping_rounds = 10, verbose = 0
)

preds_test <- ifelse(predict(model, dtest) > 0.5, 1, 0)
cm         <- confusionMatrix(as.factor(preds_test), as.factor(y_test))
print(cm)

f1_score <- cm$byClass["F1"]
print(paste("F1 Score:", f1_score))

## 8. Prepare Submission ----
# Preprocess test data for TF-IDF transformation
corpus_test <- Corpus(VectorSource(test_df$text))
dtm_test <- DocumentTermMatrix(corpus_test, control = list(dictionary = Terms(dtm_train)))

# Combine train and test DTM to ensure feature alignment (Fix for feature mismatch)
combined_dtm <- rbind(dtm_train, dtm_test)
matrix_dtm_test <- as.matrix(combined_dtm[(nrow(dtm_train) + 1):nrow(combined_dtm), ])

# Compute TF-IDF using the training IDF
tfidf_matrix_test <- sweep(matrix_dtm_test, 2, idf_vec, '*')
dtest_new <- xgb.DMatrix(data = as.matrix(tfidf_matrix_test))

# Make predictions
preds_new <- predict(model, dtest_new)
preds_binary_new <- ifelse(preds_new > 0.5, 1, 0)

# Create submission file
submission <- data.frame(id = test_df$id, target = preds_binary_new)
write.csv(submission, "submission4.csv", row.names = FALSE)