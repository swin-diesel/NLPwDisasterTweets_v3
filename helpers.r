suppressPackageStartupMessages({
  library(caret)
  library(caTools)
  library(ggplot2)
  library(stringr)
  library(textstem)
  library(tm)
  library(stopwords)      # stop‑word lists
  library(tidytext)
  library(tokenizers)
  library(tidyverse)
  library(word2vec)
  library(xgboost)
})

# ---------- shared utilities ----------
clean_text <- function(text) {
  text <- tolower(text)
  text <- gsub("http[^[:space:]]*", "", text)
  text <- gsub("[^[:alpha:][:space:]]*", "", text)
  w    <- unlist(strsplit(text, " "))
  w    <- w[!(w %in% stopwords::stopwords("en"))]
  gsub("\\s+", " ", paste(w, collapse = " "))
}

lemmatize_sentence <- function(x) {
  paste(lemmatize_strings(unlist(strsplit(x, " "))), collapse = " ")
}

americanize_map <- function(path = "uk-us-spelling-list.csv") {
  spell <- read.csv(path, stringsAsFactors = FALSE)
  setNames(spell$UK, spell$US)
}

americanize_text <- function(txt, map) {
  w <- unlist(strsplit(txt, "\\s"))
  w <- ifelse(w %in% names(map), map[w], w)
  paste(w, collapse = " ")
}

# ensure test matrix has same columns (order + zeros for missing)
fix_columns <- function(mat, ref_names) {
  miss <- setdiff(ref_names, colnames(mat))
  if (length(miss))
    mat <- cbind(mat,
                 matrix(0, nrow(mat), length(miss),
                        dimnames = list(NULL, miss)))
  mat[, ref_names, drop = FALSE]
}

# ---------- model pipelines ----------
run_v7 <- function(train_df, test_df) {
  train_df$text <- sapply(train_df$text, clean_text)
  test_df$text  <- sapply(test_df$text,  clean_text)
  
  train_df$text <- sapply(train_df$text, lemmatize_sentence)
  test_df$text  <- sapply(test_df$text, lemmatize_sentence)
  
  idx  <- sample(seq_len(nrow(train_df)), 0.80 * nrow(train_df))
  tr   <- train_df[idx, ]; te <- train_df[-idx, ]
  
  corpus <- Corpus(VectorSource(tr$text))
  dtm    <- DocumentTermMatrix(corpus)
  M      <- as.matrix(dtm)                       # training TF counts
  
  idf <- log2(nrow(M) / pmax(1, colSums(M > 0)))
  tfidf_tr <- sweep(M, 2, idf, "*")
  
  ctrl <- list(dictionary = Terms(dtm))
  dtm_te <- DocumentTermMatrix(Corpus(VectorSource(te$text)), control = ctrl)
  tfidf_te <- sweep(as.matrix(dtm_te), 2, idf, "*")
  tfidf_te <- fix_columns(tfidf_te, colnames(tfidf_tr))   # <<< key fix
  
  dtr <- xgb.DMatrix(tfidf_tr, label = tr$target)
  dte <- xgb.DMatrix(tfidf_te, label = te$target)
  
  spw <- sum(tr$target == 0) / sum(tr$target == 1)
  model <- xgb.train(list(booster="gbtree", objective="binary:logistic",
                          eval_metric="logloss", eta=0.3, max_depth=6,
                          scale_pos_weight=spw),
                     dtr, nrounds=100, verbose = 0)
  
  preds <- ifelse(predict(model, dte) > 0.5, 1, 0)
  cm    <- caret::confusionMatrix(as.factor(preds), as.factor(te$target))
  as.numeric(cm$byClass["F1"])
}

run_v8 <- function(train_df, test_df) {
  map <- americanize_map()
  train_df$text <- sapply(train_df$text, americanize_text, map)
  test_df$text  <- sapply(test_df$text,  americanize_text, map)
  run_v7(train_df, test_df)
}

run_v9 <- function(train_df, test_df) {
  map <- americanize_map()
  train_df$text <- sapply(train_df$text, americanize_text, map)
  test_df$text  <- sapply(test_df$text,  americanize_text, map)
  
  train_df$text <- sapply(train_df$text, clean_text)
  test_df$text  <- sapply(test_df$text,  clean_text)
  
  train_df$text <- sapply(train_df$text, lemmatize_sentence)
  test_df$text  <- sapply(test_df$text, lemmatize_sentence)
  
  idx  <- sample(seq_len(nrow(train_df)), 0.80 * nrow(train_df))
  tr   <- train_df[idx, ]; te <- train_df[-idx, ]
  
  w2v <- word2vec(tr$text, type="cbow", dim=100, window=5,
                  iter=10, lr=0.05, negative=5, min_count=5,
                  threads = max(1, parallel::detectCores()-1))
  D <- attr(w2v, "dim"); if (is.null(D)) D <- 100
  
  embed <- function(s, m) {
    w <- strsplit(s, " ")[[1]]
    if (!length(w)) return(rep(0, D))
    colMeans(do.call(rbind, lapply(w, function(t)
      tryCatch(predict(m, t, type="embedding"),
               error = function(e) rep(0, D)))), na.rm = TRUE)
  }
  
  trX <- t(vapply(tr$text, embed, numeric(D), m = w2v))
  teX <- t(vapply(te$text, embed, numeric(D), m = w2v))
  
  dtr <- xgb.DMatrix(trX, label = tr$target)
  dte <- xgb.DMatrix(teX, label = te$target)
  
  spw <- sum(tr$target == 0) / sum(tr$target == 1)
  model <- xgb.train(list(booster="gbtree", objective="binary:logistic",
                          eval_metric="logloss", eta=0.3, max_depth=6,
                          scale_pos_weight=spw),
                     dtr, nrounds=100, verbose = 0)
  
  preds <- ifelse(predict(model, dte) > 0.5, 1, 0)
  cm    <- caret::confusionMatrix(as.factor(preds), as.factor(te$target))
  as.numeric(cm$byClass["F1"])
}