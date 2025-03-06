# ========================================================
# Natural Language Processing with Disaster Tweets - v8
# Model: TF-IDF + XGBoost + Additional Text Cleaning
# ========================================================

## 1. Load Required Libraries ----
library(caret)        # Machine learning utilities
library(caTools)      # Data splitting
library(ggplot2)      # Data visualization
library(stringr)      # String operations
library(textstem)     # Lemmatization
library(tm)           # Text processing (Corpus, DTM)
library(tidytext)     # Text processing utilities
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

## 7. Feature Engineering (TF-IDF) ----
corpus <- Corpus(VectorSource(train_df$text))
dtm <- DocumentTermMatrix(corpus)
matrix_dtm <- as.matrix(dtm)

# Compute TF-IDF
idf <- log2(nrow(matrix_dtm) / rowSums(matrix_dtm > 0))
tfidf_matrix <- sweep(matrix_dtm, 2, idf, '*')

## 8. Model Training (XGBoost) ----
set.seed(123)

# Train-test split
split <- sample.split(train_df$target, SplitRatio = 0.8)
train_data <- tfidf_matrix[split, ]
valid_data <- tfidf_matrix[!split, ]
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
# Preprocess test data for TF-IDF transformation
corpus_test <- Corpus(VectorSource(test_df$text))
dtm_test <- DocumentTermMatrix(corpus_test, control = list(dictionary = Terms(dtm)))

# Combine train and test DTM to ensure feature alignment
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
write.csv(submission, "submission5.csv", row.names = FALSE)