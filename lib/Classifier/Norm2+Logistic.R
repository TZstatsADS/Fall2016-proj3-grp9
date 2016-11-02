library(xgboost)
#feature data 1
data<- read.csv("/Users/Jesserina/Desktop/data_norm2.csv")
data = data[,-1]
data[,1] <- as.integer(grepl("chicken",data[,1]))

set.seed(1234)
train_index <- sample(1:nrow(data),0.8*nrow(data))
train <- data[train_index, ]
test <- data[-train_index, ]

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
#dtest <- xgb.DMatrix(as.matrix(test), label = test_class)
#param <- list(max.depth = 2, eta = 1, silent = 1)

#k folds-cross validation
set.seed(111111)
nrows <- dim(train)[1]
#randomize
motorDataVld <- train[sample(1:nrows), ]
kfold <- 10

splitIndex <- (1:nrows)%%kfold
splitFactor <- factor(splitIndex[order(splitIndex)])

motorDataSub <- split(motorDataVld,splitFactor)
print(dim(motorDataSub[[1]]))
resp <- NULL

system.time({
  
for(iValid in seq(1,kfold)) {
  trainData <- NULL
  validData <- NULL
  for(j in seq(1,kfold)) {
    if(j!=iValid){
      trainData <- rbind(trainData,motorDataSub[[j]])
    }
    else {
      validData <- motorDataSub[[j]]
    }
  }
  bst <- xgboost(data = as.matrix(trainData[,-1]), label = trainData[,1], max.depth = 2, eta = 1,  nround = 2, objective = "binary:logistic")
  pred <- predict(bst, as.matrix(validData))
  p <- ifelse(pred > 0.5,1,0)
  resp<-c(resp, p)
  }
})
motorDataVld$Xpred <- resp

tb <- table(motorDataVld[,1],motorDataVld$Xpred)
tb


error <- 1 - sum(diag(tb))/sum(tb)
error

