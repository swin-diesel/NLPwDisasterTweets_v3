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
test_df  <- read.csv("test.csv",  stringsAsFactors = FALSE)

## 3. Exploratory Data Analysis ----
print(head(train_df))
print(summary(train_df))

ggplot(train_df, aes(x = factor(target))) +
  geom_bar(fill = c("red", "blue")) +
  labs(title = "Distribution of Disaster vs. Non‑Disaster Tweets",
       x = "Tweet Type", y = "Count") +
  scale_x_discrete(labels = c("Non‑Disaster", "Disaster"))

## 4. Custom Text Cleaning ----
spellings  <- read.csv("uk-us-spelling-list.csv", stringsAsFactors = FALSE)
us_to_uk   <- setNames(spellings$UK, spellings$US)
uk_to_us   <- setNames(spellings$US, spellings$UK)

americanize <- function(text) {
  w <- unlist(strsplit(text, "\\s"))
  w <- ifelse(w %in% names(uk_to_us), uk_to_us[w], w)
  paste(w, collapse = " ")
}

anglicize <- function(text) {
  w <- unlist(strsplit(text, "\\s"))
  w <- ifelse(w %in% names(us_to_uk), us_to_uk[w], w)
  paste(w, collapse = " ")
}

train_df$text <- sapply(train_df$text, americanize)

## 5. General Text Preprocessing ----
clean_text <- function(text) {
  text <- tolower(text)
  text <- gsub("http[^[:space:]]*", "", text)          # URLs
  text <- gsub("[^[:alpha:][:space:]]*", "", text)     # Punct & digits
  w    <- unlist(strsplit(text, " "))
  w    <- w[!(w %in% stopwords("en"))]                 # Stop‑words
  gsub("\\s+", " ", paste(w, collapse = " "))          # Collapse spaces
}

train_df$text <- sapply(train_df$text, clean_text)
test_df$text  <- sapply(test_df$text,  clean_text)

## 6. Tokenization & Lemmatization ----
train_df$text <- sapply(
  lapply(strsplit(train_df$text, " "), lemmatize_strings), paste, collapse = " "
)
test_df$text  <- sapply(
  lapply(strsplit(test_df$text,  " "), lemmatize_strings), paste, collapse = " "
)

## 7. Feature Engineering (TF‑IDF) ----------
set.seed(123)

# Outer 80/20 split (train_val vs hold‑out test)
train_idx   <- sample(seq_len(nrow(train_df)), 0.80 * nrow(train_df))
train_split <- train_df[train_idx, ]
test_split  <- train_df[-train_idx, ]          # internal hold‑out

# Inner 80/20 split on train_split for early‑stopping
inner_idx   <- sample(seq_len(nrow(train_split)), 0.80 * nrow(train_split))
train_inner <- train_split[inner_idx, ]
val_inner   <- train_split[-inner_idx, ]

# DTM on train_inner only
corpus_train <- Corpus(VectorSource(train_inner$text))
dtm_train    <- DocumentTermMatrix(corpus_train)
mat_train    <- as.matrix(dtm_train)

# Column‑wise IDF
idf_vec     <- log2(nrow(mat_train) / pmax(1, colSums(mat_train > 0)))
tfidf_train <- sweep(mat_train, 2, idf_vec, "*")

# Helper for column alignment
fix_columns <- function(mat, ref_names) {
  miss <- setdiff(ref_names, colnames(mat))
  if (length(miss))
    mat <- cbind(mat,
                 matrix(0, nrow(mat), length(miss),
                        dimnames = list(NULL, miss)))
  mat[, ref_names]
}

# Validation & internal‑test matrices
ctrl       <- list(dictionary = Terms(dtm_train))
mat_val    <- as.matrix(DocumentTermMatrix(
  Corpus(VectorSource(val_inner$text)),  control = ctrl))
mat_test   <- as.matrix(DocumentTermMatrix(
  Corpus(VectorSource(test_split$text)), control = ctrl))

mat_val  <- fix_columns(mat_val,  colnames(mat_train))
mat_test <- fix_columns(mat_test, colnames(mat_train))

tfidf_val  <- sweep(mat_val,  2, idf_vec, "*")
tfidf_test <- sweep(mat_test, 2, idf_vec, "*")

y_train <- train_inner$target
y_val   <- val_inner$target
y_test  <- test_split$target

## 8. Model Training (XGBoost) ----
dtrain <- xgb.DMatrix(tfidf_train, label = y_train)
dval   <- xgb.DMatrix(tfidf_val,   label = y_val)
dtest  <- xgb.DMatrix(tfidf_test,  label = y_test)

params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.3,
  max_depth = 6,
  scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
)

model <- xgb.train(
  params, dtrain, nrounds = 100,
  watchlist = list(train = dtrain, val = dval),
  early_stopping_rounds = 10, verbose = 0
)

preds_test <- ifelse(predict(model, dtest) > 0.5, 1, 0)
cm         <- confusionMatrix(as.factor(preds_test), as.factor(y_test))
print(cm)
print(paste("F1 Score:", cm$byClass["F1"]))

## 9. Prepare Submission ----
corpus_test <- Corpus(VectorSource(test_df$text))
dtm_test <- DocumentTermMatrix(
  corpus_test, control = list(dictionary = Terms(dtm_train))
)

combined_dtm     <- rbind(dtm_train, dtm_test)
matrix_dtm_test  <- as.matrix(
  combined_dtm[(nrow(dtm_train) + 1):nrow(combined_dtm), ]
)

tfidf_matrix_test <- sweep(matrix_dtm_test, 2, idf_vec, "*")
dtest_new <- xgb.DMatrix(tfidf_matrix_test)

preds_submit <- predict(model, dtest_new)
submission   <- data.frame(
  id     = test_df$id,
  target = ifelse(preds_submit > 0.5, 1, 0)
)
write.csv(submission, "submission5.csv", row.names = FALSE)