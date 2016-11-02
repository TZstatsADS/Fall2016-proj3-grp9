library(kernlab)
library(e1071)
library(rpart)
library(class)
library(gbm)

### Build the Test function 
Test = function(new.data){
  norm2_test = new.data
  ####cleaning data####
  norm2_test <- subset(norm2_test, select=-c(X,X0))
  x_test <- norm2_test
  
  ##### PCA #########
  pca_test <- prcomp(x_test,scale = T)
  
  # select the number of PC
  pca_xtest = pca_test$x[,1:5]
  
  ######Testing Model Performance for SVM#####
  pred_test <- predict(model$svm.adv,pca_xtest)
  for (i in 1:2000){
    if (pred_test[i] > 0.5){
      norm2_test$pred_Categor_svm[i] = 1
    }
    else{
      norm2_test$pred_Categor_svm[i] = 0
    }
  } 
  
  ######Testing Model Performance for GBM#####
  pred_test_gbm <- predict(model$gbm.base,pca_xtest,n.trees = 100)
  for (i in 1:2000){
    if (pred_test_gbm[i] > 0){
      norm2_test$pred_Categor_gbm[i] = 1
    }
    else{
      norm2_test$pred_Categor_gbm[i] = 0
    }
  } 
  results = as.data.frame(cbind(norm2_test$pred_Categor_svm,norm2_test$pred_Categor_gbm))
  names(results) = c("svm.adv","gbm.base")
  return = results
}

newdat = read.csv(file.choose(), header = T)
result = Test(newdat)
