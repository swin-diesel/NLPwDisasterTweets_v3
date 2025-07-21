# =======================================================
#  NLP with Disaster Tweets – Combined Driver (v7 | v8 | v9)
#  Each pipeline runs in its own local environment, prints its F1,
#  and appends to a summary table.  No submission files are written.
# =======================================================

cat("\nLoading packages …\n")
suppressPackageStartupMessages({
  library(caret); library(caTools); library(ggplot2); library(stringr)
  library(textstem); library(tm); library(tidytext); library(tokenizers)
  library(tidyverse); library(word2vec); library(xgboost)
})

train_master <- read.csv("train.csv", stringsAsFactors = FALSE)
test_master  <- read.csv("test.csv",  stringsAsFactors = FALSE)

f1_scores <- list()

# -------------------------------------------------------
#  v7  TF‑IDF baseline
# -------------------------------------------------------
local({
  
  cat("\n=== v7 TF‑IDF baseline =========================\n")
  
  train_df <- train_master
  test_df  <- test_master
  
  # ---- text prep ---------------------------------------------------
  clean_text <- function(text) {
    text <- tolower(text)
    text <- gsub("http[^[:space:]]*", "", text)
    text <- gsub("[^[:alpha:][:space:]]*", "", text)
    w    <- unlist(strsplit(text, " "))
    w    <- w[!(w %in% stopwords("en"))]
    paste(w, collapse = " ")
  }
  train_df$text <- sapply(train_df$text, clean_text)
  test_df$text  <- sapply(test_df$text,  clean_text)
  
  train_df$text <- sapply(
    lapply(strsplit(train_df$text, " "), lemmatize_strings), paste, collapse = " "
  )
  test_df$text  <- sapply(
    lapply(strsplit(test_df$text,  " "), lemmatize_strings), paste, collapse = " "
  )
  
  # ---- splits ------------------------------------------------------
  set.seed(123)
  train_idx   <- sample(seq_len(nrow(train_df)), 0.80 * nrow(train_df))
  train_split <- train_df[train_idx, ]
  test_split  <- train_df[-train_idx, ]
  
  inner_idx   <- sample(seq_len(nrow(train_split)), 0.80 * nrow(train_split))
  train_inner <- train_split[inner_idx, ]
  val_inner   <- train_split[-inner_idx, ]
  
  # ---- TF‑IDF ------------------------------------------------------
  corpus_train <- Corpus(VectorSource(train_inner$text))
  dtm_train    <- DocumentTermMatrix(corpus_train)
  mat_train    <- as.matrix(dtm_train)
  
  idf_vec      <- log2(nrow(mat_train) / pmax(1, colSums(mat_train > 0)))
  tfidf_train  <- sweep(mat_train, 2, idf_vec, "*")
  
  fix_columns <- function(mat, ref_names) {
    miss <- setdiff(ref_names, colnames(mat))
    if (length(miss))
      mat <- cbind(mat,
                   matrix(0, nrow(mat), length(miss),
                          dimnames = list(NULL, miss)))
    mat[, ref_names]
  }
  
  ctrl     <- list(dictionary = Terms(dtm_train))
  mat_val  <- as.matrix(DocumentTermMatrix(
    Corpus(VectorSource(val_inner$text)),  control = ctrl))
  mat_test <- as.matrix(DocumentTermMatrix(
    Corpus(VectorSource(test_split$text)), control = ctrl))
  
  mat_val  <- fix_columns(mat_val,  colnames(mat_train))
  mat_test <- fix_columns(mat_test, colnames(mat_train))
  
  tfidf_val  <- sweep(mat_val,  2, idf_vec, "*")
  tfidf_test <- sweep(mat_test, 2, idf_vec, "*")
  
  # ---- XGBoost -----------------------------------------------------
  dtrain <- xgb.DMatrix(tfidf_train, label = train_inner$target)
  dval   <- xgb.DMatrix(tfidf_val,   label = val_inner$target)
  dtest  <- xgb.DMatrix(tfidf_test,  label = test_split$target)
  
  params <- list(
    booster = "gbtree", objective = "binary:logistic", eval_metric = "logloss",
    eta = 0.3, max_depth = 6,
    scale_pos_weight = sum(train_inner$target == 0) / sum(train_inner$target == 1)
  )
  
  model <- xgb.train(
    params, dtrain, nrounds = 100,
    watchlist = list(train = dtrain, val = dval),
    early_stopping_rounds = 10, verbose = 0
  )
  
  preds <- ifelse(predict(model, dtest) > 0.5, 1, 0)
  cm    <- confusionMatrix(as.factor(preds), as.factor(test_split$target))
  f1_scores$v7 <<- as.numeric(cm$byClass["F1"])
  cat("F1 =", round(f1_scores$v7, 3), "\n")
})

# -------------------------------------------------------
#  v8  Enhanced cleaning + TF‑IDF
# -------------------------------------------------------
local({
  
  cat("\n=== v8 Clean TF‑IDF =================================\n")
  
  train_df <- train_master
  test_df  <- test_master
  
  spellings <- read.csv("uk-us-spelling-list.csv", stringsAsFactors = FALSE)
  us_to_uk  <- setNames(spellings$UK, spellings$US)
  
  americanize <- function(txt) {
    w <- unlist(strsplit(txt, "\\s"))
    w <- ifelse(w %in% names(us_to_uk), us_to_uk[w], w)
    paste(w, collapse = " ")
  }
  train_df$text <- sapply(train_df$text, americanize)
  
  clean_text <- function(text) {
    text <- tolower(text)
    text <- gsub("http[^[:space:]]*", "", text)
    text <- gsub("[^[:alpha:][:space:]]*", "", text)
    w    <- unlist(strsplit(text, " "))
    w    <- w[!(w %in% stopwords("en"))]
    gsub("\\s+", " ", paste(w, collapse = " "))
  }
  train_df$text <- sapply(train_df$text, clean_text)
  test_df$text  <- sapply(test_df$text,  clean_text)
  
  train_df$text <- sapply(
    lapply(strsplit(train_df$text, " "), lemmatize_strings), paste, collapse = " "
  )
  test_df$text  <- sapply(
    lapply(strsplit(test_df$text,  " "), lemmatize_strings), paste, collapse = " "
  )
  
  # splits
  set.seed(123)
  train_idx   <- sample(seq_len(nrow(train_df)), 0.80 * nrow(train_df))
  train_split <- train_df[train_idx, ]
  test_split  <- train_df[-train_idx, ]
  
  inner_idx   <- sample(seq_len(nrow(train_split)), 0.80 * nrow(train_split))
  train_inner <- train_split[inner_idx, ]
  val_inner   <- train_split[-inner_idx, ]
  
  # TF‑IDF
  corpus_train <- Corpus(VectorSource(train_inner$text))
  dtm_train    <- DocumentTermMatrix(corpus_train)
  mat_train    <- as.matrix(dtm_train)
  
  idf_vec     <- log2(nrow(mat_train) / pmax(1, colSums(mat_train > 0)))
  tfidf_train <- sweep(mat_train, 2, idf_vec, "*")
  
  fix_columns <- function(mat, ref_names) {
    miss <- setdiff(ref_names, colnames(mat))
    if (length(miss))
      mat <- cbind(mat,
                   matrix(0, nrow(mat), length(miss),
                          dimnames = list(NULL, miss)))
    mat[, ref_names]
  }
  
  ctrl      <- list(dictionary = Terms(dtm_train))
  mat_val   <- as.matrix(DocumentTermMatrix(
    Corpus(VectorSource(val_inner$text)),  control = ctrl))
  mat_test  <- as.matrix(DocumentTermMatrix(
    Corpus(VectorSource(test_split$text)), control = ctrl))
  
  mat_val  <- fix_columns(mat_val,  colnames(mat_train))
  mat_test <- fix_columns(mat_test, colnames(mat_train))
  
  tfidf_val  <- sweep(mat_val,  2, idf_vec, "*")
  tfidf_test <- sweep(mat_test, 2, idf_vec, "*")
  
  dtrain <- xgb.DMatrix(tfidf_train, label = train_inner$target)
  dval   <- xgb.DMatrix(tfidf_val,   label = val_inner$target)
  dtest  <- xgb.DMatrix(tfidf_test,  label = test_split$target)
  
  params <- list(
    booster = "gbtree", objective = "binary:logistic", eval_metric = "logloss",
    eta = 0.3, max_depth = 6,
    scale_pos_weight = sum(train_inner$target == 0) / sum(train_inner$target == 1)
  )
  
  model <- xgb.train(
    params, dtrain, nrounds = 100,
    watchlist = list(train = dtrain, val = dval),
    early_stopping_rounds = 10, verbose = 0
  )
  
  preds <- ifelse(predict(model, dtest) > 0.5, 1, 0)
  cm    <- confusionMatrix(as.factor(preds), as.factor(test_split$target))
  f1_scores$v8 <<- as.numeric(cm$byClass["F1"])
  cat("F1 =", round(f1_scores$v8, 3), "\n")
})

# -------------------------------------------------------
#  v9  Word2Vec embeddings
# -------------------------------------------------------
local({
  
  cat("\n=== v9 Word2Vec =====================================\n")
  
  train_df <- train_master
  test_df  <- test_master
  
  # same spelling map from env_v8
  americanize <- function(t) {
    w <- unlist(strsplit(t, "\\s"))
    w <- ifelse(w %in% names(us_to_uk), us_to_uk[w], w)
    paste(w, collapse = " ")
  }
  train_df$text <- sapply(train_df$text, americanize)
  
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
  
  train_df$text <- sapply(
    lapply(strsplit(train_df$text, " "), lemmatize_strings), paste, collapse = " "
  )
  test_df$text  <- sapply(
    lapply(strsplit(test_df$text,  " "), lemmatize_strings), paste, collapse = " "
  )
  
  # splits
  set.seed(123)
  train_idx   <- sample(seq_len(nrow(train_df)), 0.80 * nrow(train_df))
  train_split <- train_df[train_idx, ]
  test_split  <- train_df[-train_idx, ]
  
  inner_idx   <- sample(seq_len(nrow(train_split)), 0.80 * nrow(train_split))
  train_inner <- train_split[inner_idx, ]
  val_inner   <- train_split[-inner_idx, ]
  
  # Word2Vec
  w2v <- word2vec(
    x=train_inner$text, type="cbow", dim=100, window=5, iter=10,
    lr=0.05, negative=5, min_count=5,
    threads = parallel::detectCores()-1, encoding="UTF-8"
  )
  embed_dim <- attr(w2v, "dim")
  if (is.null(embed_dim)) embed_dim <- 100
  
  get_embed <- function(sent, model){
    w <- unlist(strsplit(sent," "))
    if (!length(w)) return(rep(0, embed_dim))
    vecs <- t(sapply(w, function(word){
      tryCatch(predict(model, word, type="embedding"),
               error=function(e) rep(0, embed_dim))
    }))
    colMeans(vecs, na.rm = TRUE)
  }
  
  embed_train <- t(sapply(train_inner$text, get_embed, model=w2v))
  embed_val   <- t(sapply(val_inner$text,   get_embed, model=w2v))
  embed_test  <- t(sapply(test_split$text,  get_embed, model=w2v))
  
  dtrain <- xgb.DMatrix(embed_train, label = train_inner$target)
  dval   <- xgb.DMatrix(embed_val,   label = val_inner$target)
  dtest  <- xgb.DMatrix(embed_test,  label = test_split$target)
  
  params <- list(
    booster="gbtree", objective="binary:logistic", eval_metric="logloss",
    eta=0.3, max_depth=6,
    scale_pos_weight = sum(train_inner$target==0)/sum(train_inner$target==1)
  )
  
  model <- xgb.train(
    params, dtrain, nrounds=100,
    watchlist = list(train=dtrain, val=dval),
    early_stopping_rounds=10, verbose=0
  )
  
  preds <- ifelse(predict(model, dtest) > 0.5, 1, 0)
  cm    <- confusionMatrix(as.factor(preds), as.factor(test_split$target))
  f1_scores$v9 <<- as.numeric(cm$byClass["F1"])
  cat("F1 =", round(f1_scores$v9, 3), "\n")
})

# -------------------------------------------------------
#  Summary
# -------------------------------------------------------
cat("\n=========== F1 comparison ===========\n")
summary_tbl <- data.frame(
  Pipeline = c("v7 TF‑IDF", "v8 Clean TF‑IDF", "v9 Word2Vec"),
  F1_Score = unlist(f1_scores)
)
print(summary_tbl, row.names = FALSE)