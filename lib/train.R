library(kernlab)
library(e1071)
library(rpart)
library(class)
library(gbm)

### Build the Trining function 
Train = function(data, Y){
  data$Y[1:1000] = 1
  data$Y[1001:2000] = 0
  norm2 = data
  ####cleaning data####
  norm2 <- subset(norm2, select=-c(X,X0))
  x <- subset(norm2, select = -Y)
  y <- subset(norm2, select = Y)
  
  ##### PCA #########
  pca <- prcomp(x,scale = T)
  
  #select number of PC# 
  pca_x = pca$x[,1:5]
  
  ### svm classification after PCA
  # cross validation of 20 fold
  tc <- tune.control(cross = 20)
  Cs = c(.001,.01,.1,.5,1)
  gammas = 10^(-2:2)
  degres = c(0,1)
  coef = c(0,1)
  cv_svmTune <- tune.svm(pca_x, y =y, cost = Cs, gamma = gammas,degree = degres,coef0 = coef,
                         tunecontrol = tc)
  summary(cv_svmTune)
  ###### best adv model 
  p = cv_svmTune$best.parameters
  best.svm = svm(pca_x,y,cost = p[,4], gamma = p[,2],degree = p[,1],coef0 = p[,3])
  
  ####baseline model###
  gbm_model <- gbm.fit(pca_x,as.vector(as.matrix(y)),distribution = "bernoulli", n.trees = 100)
  return(list(gbm.base=gbm_model,svm.adv=best.svm))
  
}

### model trainning
# read training feature data
norm2 = read.csv(file.choose(), header = T)
# add label on feature data
norm2$Num_Categories[1:1000] = 1
norm2$Num_Categories[1001:2000] = 0
# train model
model = Train(data = norm2, Y = Num_Categories)