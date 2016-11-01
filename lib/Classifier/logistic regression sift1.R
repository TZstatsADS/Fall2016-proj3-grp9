library(xgboost)

data <- read.csv("/Users/Jesserina/Desktop/SIFT1T.csv")
data$Num_Categories <- NULL
data$Categories <- NULL

labels <- gsub("[_[:digit:]]","", data[,1])

class <- ifelse(labels == "dog", 1, 0)

X <- data[,-1]

set.seed(1234)
train_index <- sample(1:nrow(data),0.8*nrow(data))
train <- X[train_index, ]
test <- X[-train_index, ]
train_class <- class[train_index]
test_class <- class[-train_index]

#eXtreme Gradient Boosting,s an efficient and scalable implementation of gradient boosting framework
# Can deal with Sparse Matrix

#eta: step size of each boosting step
#max.depth: maximum depth of the tree
#nround: the max number of iterations

# #logistic regression classifier
# bst <- xgboost(data = train_class, label = train, max.depth = 2, eta = 1,  nround = 2, objective = "binary:logistic")
# 
# xgb.save(bst, 'model.save')
# pred <- predict(bst, test$data)
# 
# #xgb.dump(bst, 'model.dump')

dtrain <- xgb.DMatrix( as.matrix(train), label = train_class)
#class(dtrain)
#head(getinfo(dtrain,'label'))
# xgb.DMatrix.save(dtrain, 'xgb.DMatrix')
# dtrain = xgb.DMatrix('xgb.DMatrix')
logregobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1/(1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}


evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1/(1 + exp(-preds))
  preds_class <- ifelse(preds > 0.5, 1, 0)
  tb <- table(preds_class,labels)
  err <- 1 - sum(diag(tb))/sum(tb)
  return(list(metric = "Mistake_class_rate", value = err))
}
dtest <- xgb.DMatrix(as.matrix(test), label = test_class)
watchlist <- list(eval = dtest, train = dtrain)
param <- list(max.depth = 2, eta = 1, silent = 1)

bst <- xgb.train(param, dtrain, nround = 30, watchlist, logregobj , evalerror)

bst

