# Natural Language Processing with Disaster Tweets_v4

## Preliminary Tasks

#Load libraries
library(caret)
library(caTools)
library(ggplot2)
library(stringr)
library(textstem)
library(tidytext)
library(tm)
library(word2vec)
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

## Feature Engineering with BERT

# Load the 'reticulate' package
library(reticulate)

# Import the 'transformers' library
transformers <- import("transformers")

# Import the specific BERT model for sequence classification
BertForSequenceClassification <- transformers$BertForSequenceClassification

# Load the pre-trained BERT model using the correct method call
bert_model <- BertForSequenceClassification$from_pretrained("bert-base-uncased")

# Load the BERT tokenizer
tokenizer <- transformers$BertTokenizer$from_pretrained("bert-base-uncased")

# Function to generate BERT embeddings for a sentence
get_sentence_embedding <- function(sentence, model, tokenizer) {
  encoded_inputs <- tokenizer$encode_plus(
    sentence, 
    add_special_tokens = TRUE, 
    max_length = as.integer(512), 
    padding = "max_length", 
    truncation = TRUE, 
    return_attention_mask = TRUE, 
    return_tensors = "pt"
  )
  
  embeddings <- model(encoded_inputs$input_ids, attention_mask = encoded_inputs$attention_mask)
  last_hidden_state <- embeddings$last_hidden_state
  
  # Check if last_hidden_state is a valid tensor and calculate the average vector
  if (!is.null(last_hidden_state) && py_is_instance(last_hidden_state, "torch.Tensor")) {
    avg_vector <- reticulate::py_run_string("
import torch
avg_vector = torch.mean(last_hidden_state, dim=1).numpy()
", local = list(last_hidden_state = last_hidden_state), convert = TRUE)$avg_vector
  } else {
    # Return a default numeric vector of length 768 when embedding fails
    avg_vector <- rep(NA, 768)
  }
  
  if (is.null(avg_vector)) {
    stop("Failed to calculate the average vector.")
  }
  
  return(avg_vector)
}

# Function to process batches of texts
process_batch <- function(texts, model, tokenizer) {
  embeddings <- sapply(texts, get_sentence_embedding, model = model, tokenizer = tokenizer)
  return(t(embeddings))  # Transpose to match expected format
}

# Define batch size
batch_size <- 5  # Adjust based on your system's capability

# Generate BERT embeddings for the training data in batches
train_embeddings <- list()
for (i in seq(1, nrow(train_df), by = batch_size)) {
  batch_texts <- train_df$text[i:min(i + batch_size - 1, nrow(train_df))]
  train_embeddings[[length(train_embeddings) + 1]] <- process_batch(batch_texts, bert_model, tokenizer)
  # Optionally clear variables to free up memory
  rm(batch_texts)
  gc()  # Force garbage collection
}
train_embeddings <- do.call(rbind, train_embeddings)

# Generate BERT embeddings for the test data in batches
test_embeddings <- list()
for (i in seq(1, nrow(test_df), by = batch_size)) {
  batch_texts <- test_df$text[i:min(i + batch_size - 1, nrow(test_df))]
  test_embeddings[[length(test_embeddings) + 1]] <- process_batch(batch_texts, bert_model, tokenizer)
  # Optionally clear variables to free up memory
  rm(batch_texts)
  gc()  # Force garbage collection
}
test_embeddings <- do.call(rbind, test_embeddings)

## Build the Model: Gradient Boosting

# Setting seed for reproducibility
set.seed(123)

# Partition Data
split <- sample.split(train_df$target, SplitRatio = 0.8)
train_data <- train_embeddings[split, ]
valid_data <- train_embeddings[!split, ]
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
dtest_new <- xgb.DMatrix(data = as.matrix(test_embeddings))
preds_new <- predict(model, dtest_new)
preds_binary_new <- ifelse(preds_new > 0.5, 1, 0)

# Create submission dataframe
submission <- data.frame(id = test_df$id, target = preds_binary_new)

# Write submission to CSV
write.csv(submission, "submission3.csv", row.names = FALSE)