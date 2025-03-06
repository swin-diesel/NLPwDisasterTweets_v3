# ========================================================
# Natural Language Processing with Disaster Tweets - v9
# Model: Word2Vec + XGBoost
# ========================================================

## 1. Load Required Libraries ----
library(caret)        # Machine learning utilities
library(caTools)      # Data splitting
library(ggplot2)      # Data visualization
library(stringr)      # String operations
library(textstem)     # Lemmatization
library(tidytext)     # Text processing utilities
library(tm)           # Text processing (Corpus, DTM)
library(word2vec)     # Word2Vec embeddings
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

## 4. Custom Text Cleaning ----

# Load dictionary for American/British spelling conversion
spellings <- read.csv("uk-us-spelling-list.csv", stringsAsFactors = FALSE)
us_to_uk <- setNames(spellings$UK, spellings$US)
uk_to_us <- setNames(spellings$US, spellings$UK)

# Function to convert US-English to UK-English
americanize <- function(text) {
  words <- unlist(strsplit(text, "\\s"))
  words <- ifelse(words %in% names(uk_to_us), uk_to_us[words], words)
  return(paste(words, collapse = " "))
}

# Function to convert UK-English to US-English
anglicize <- function(text) {
  words <- unlist(strsplit(text, "\\s"))
  words <- ifelse(words %in% names(us_to_uk), us_to_uk[words], words)
  return(paste(words, collapse = " "))
}

# Apply Americanization to training text
train_df$text <- sapply(train_df$text, americanize)

## 5. General Text Preprocessing ----
clean_text <- function(text) {
  text <- tolower(text)
  text <- gsub("http[^[:space:]]*", "", text)  # Remove URLs
  text <- gsub("[^[:alpha:][:space:]]*", "", text)  # Remove punctuation & numbers
  words <- unlist(strsplit(text, " "))
  words <- words[!(words %in% stopwords("en"))]  # Remove stopwords
  text <- paste(words, collapse = " ")
  return(gsub("\\s+", " ", text))  # Remove extra whitespace
}

# Apply cleaning function
train_df$text <- sapply(train_df$text, clean_text)
test_df$text <- sapply(test_df$text, clean_text)

## 6. Tokenization & Lemmatization ----
tokenized_text <- strsplit(train_df$text, " ")
lemmatized_text <- lapply(tokenized_text, lemmatize_strings)
train_df$text <- sapply(lemmatized_text, paste, collapse = " ")

tokenized_test_text <- strsplit(test_df$text, " ")
lemmatized_test_text <- lapply(tokenized_test_text, lemmatize_strings)
test_df$text <- sapply(lemmatized_test_text, paste, collapse = " ")

## 7. Feature Engineering with Word2Vec ----

# Train word2vec model
word2vec_model <- word2vec(
  x = train_df$text,
  type = "cbow",
  dim = 100,
  window = 5,
  iter = 10,
  lr = 0.05,
  hs = FALSE,
  negative = 5,
  sample = 0.001,
  min_count = 5,
  split = c(" \n,.-!?:;/\"#$%&'()*+<=>@[]\\^_`{|}~\t\v\f\r", ".\n?!"),
  stopwords = stopwords("en"),
  threads = parallel::detectCores() - 1,
  encoding = "UTF-8"
)

# Generate embeddings for each tweet
get_sentence_embedding <- function(sentence, model) {
  words <- unlist(strsplit(sentence, " "))
  
  # Initialize a matrix to store the vectors
  vectors <- matrix(0, length(words), 100)  # Assuming the dimension of the embeddings is 100
  
  for (i in 1:length(words)) {
    # Attempt to get the vector for the word
    # If the word is not in the model's vocabulary, skip it
    tryCatch({
      vectors[i, ] <- predict(model, words[i], type = "embedding")
    }, error = function(e) {})
  }
  
  # Calculate the average vector for the sentence
  avg_vector <- colMeans(vectors, na.rm = TRUE)
  return(avg_vector)
}

train_embeddings <- t(sapply(train_df$text, get_sentence_embedding, model = word2vec_model))
test_embeddings <- t(sapply(test_df$text, get_sentence_embedding, model = word2vec_model))

## 8. Model Training (XGBoost) ----
set.seed(123)

# Train-test split
split <- sample.split(train_df$target, SplitRatio = 0.8)
train_data <- train_embeddings[split, ]
valid_data <- train_embeddings[!split, ]
train_target <- train_df$target[split]
valid_target <- train_df$target[!split]

# Convert to DMatrix format
dtrain <- xgb.DMatrix(data = as.matrix(train_data), label = train_target)
dvalid <- xgb.DMatrix(data = as.matrix(valid_data), label = valid_target)

# XGBoost parameters
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

# Predictions on validation data
preds <- predict(model, dvalid)
preds_binary <- ifelse(preds > 0.5, 1, 0)

# Evaluate the model using confusion matrix
cm <- confusionMatrix(as.factor(preds_binary), as.factor(valid_target))
print(cm)

# Extract and print F1 Score
f1_score <- cm$byClass["F1"]
print(paste("F1 Score:", f1_score))

## 9. Prepare Submission ----
# Convert test embeddings to DMatrix
dtest_new <- xgb.DMatrix(data = as.matrix(test_embeddings))

# Make predictions
preds_new <- predict(model, dtest_new)
preds_binary_new <- ifelse(preds_new > 0.5, 1, 0)

# Create submission file
submission <- data.frame(id = test_df$id, target = preds_binary_new)
write.csv(submission, "submission6.csv", row.names = FALSE)