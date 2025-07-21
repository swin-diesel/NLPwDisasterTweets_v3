# ========================================================
# Natural Language Processing with Disaster Tweets - v9
# Model: Word2Vec + XGBoost
# ========================================================

## 1. Load Required Libraries ----
library(caret)
library(caTools)
library(ggplot2)
library(stringr)
library(textstem)
library(tidytext)
library(tm)
library(word2vec)
library(xgboost)

## 2. Load Datasets ----
train_df <- read.csv("train.csv", stringsAsFactors = FALSE)
test_df  <- read.csv("test.csv",  stringsAsFactors = FALSE)

## 3. EDA (unchanged) ----
print(head(train_df)); print(summary(train_df))
ggplot(train_df, aes(x = factor(target))) +
  geom_bar(fill = c("red", "blue")) +
  labs(title = "Distribution of Disaster vs Non‑Disaster Tweets",
       x = "Tweet Type", y = "Count") +
  scale_x_discrete(labels = c("Non‑Disaster", "Disaster"))

## 4. Custom Text Cleaning ----
spellings <- read.csv("uk-us-spelling-list.csv", stringsAsFactors = FALSE)
us_to_uk  <- setNames(spellings$UK, spellings$US)
uk_to_us  <- setNames(spellings$US, spellings$UK)

americanize <- function(t) {
  w <- unlist(strsplit(t, "\\s")); w <- ifelse(w %in% names(uk_to_us), uk_to_us[w], w)
  paste(w, collapse = " ")
}
train_df$text <- sapply(train_df$text, americanize)

## 5. General Text Preprocessing ----
clean_text <- function(t) {
  t <- tolower(t)
  t <- gsub("http[^[:space:]]*", "", t)
  t <- gsub("[^[:alpha:][:space:]]*", "", t)
  w <- unlist(strsplit(t, " "))
  w <- w[!(w %in% stopwords("en"))]
  gsub("\\s+", " ", paste(w, collapse = " "))
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

## 7. Train/Validation/Test Splits ----
set.seed(123)
train_idx   <- sample(seq_len(nrow(train_df)), 0.80 * nrow(train_df))
train_split <- train_df[train_idx, ]
test_split  <- train_df[-train_idx, ]

inner_idx   <- sample(seq_len(nrow(train_split)), 0.80 * nrow(train_split))
train_inner <- train_split[inner_idx, ]
val_inner   <- train_split[-inner_idx, ]

## 8. Word2Vec Model (fit on train_inner only) ----
word2vec_model <- word2vec(
  x        = train_inner$text,
  type     = "cbow",
  dim      = 100,
  window   = 5,
  iter     = 10,
  lr       = 0.05,
  hs       = FALSE,
  negative = 5,
  sample   = 0.001,
  min_count = 5,
  split    = c(" \n,.-!?:;/\"#$%&'()*+<=>@[]\\^_`{|}~\t\v\f\r", ".\n?!"),
  stopwords = stopwords("en"),
  threads   = parallel::detectCores() - 1,
  encoding  = "UTF-8"
)

##############  CRITICAL: grab the true embedding size once  ##############
# Without this, embed_dim can be NULL, which breaks the zero-vector guard
embed_dim <- attr(word2vec_model, "dim")
if (is.null(embed_dim)) {
  embed_dim <- ncol(predict(word2vec_model, "the", type = "embedding"))
}
############################################################################

get_sentence_embedding <- function(sent, model) {
  w <- unlist(strsplit(sent, " "))
  w <- w[nchar(w) > 0]
  
  if (length(w) == 0)
    return(rep(0, embed_dim))   # 100-dim zero vector
  
  vecs <- t(sapply(w, function(word) {
    out <- tryCatch(predict(model, word, type = "embedding"),
                    error = function(e) NULL)
    if (is.null(out)) rep(0, embed_dim) else out
  }))
  colMeans(vecs, na.rm = TRUE)
}

embed_train <- t(sapply(train_inner$text, get_sentence_embedding,
                        model = word2vec_model))
embed_val   <- t(sapply(val_inner$text,   get_sentence_embedding,
                        model = word2vec_model))
embed_test  <- t(sapply(test_split$text,  get_sentence_embedding,
                        model = word2vec_model))

## 9. Model Training (XGBoost) ----
dtrain <- xgb.DMatrix(embed_train, label = train_inner$target)
dval   <- xgb.DMatrix(embed_val,   label = val_inner$target)
dtest  <- xgb.DMatrix(embed_test,  label = test_split$target)

params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.3,
  max_depth = 6,
  scale_pos_weight = sum(train_inner$target == 0) / sum(train_inner$target == 1)
)

model <- xgb.train(
  params, dtrain, nrounds = 100,
  watchlist = list(train = dtrain, val = dval),
  early_stopping_rounds = 10, verbose = 0
)

preds_test <- ifelse(predict(model, dtest) > 0.5, 1, 0)
cm <- confusionMatrix(as.factor(preds_test), as.factor(test_split$target))
print(cm); print(paste("F1 Score:", cm$byClass["F1"]))

## 10. Prepare Submission ----
embed_public <- t(sapply(test_df$text, get_sentence_embedding, model = word2vec_model))
dtest_pub    <- xgb.DMatrix(embed_public)

preds_submit <- predict(model, dtest_pub)
submission   <- data.frame(id = test_df$id,
                           target = ifelse(preds_submit > 0.5, 1, 0))
write.csv(submission, "submission6.csv", row.names = FALSE)