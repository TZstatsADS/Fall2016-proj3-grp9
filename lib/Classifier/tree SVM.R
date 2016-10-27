SIFT1 <- read.csv(file = "C:/Users/liran/OneDrive/TC Courses/ADS/Proj 3/Project3_poodleKFC_train/sift_features.csv", header = T) 
SIFT1T <- t(SIFT1)
SIFT1T <- read.csv(file = "C:/Users/liran/OneDrive/TC Courses/ADS/Proj 3/SIFT1T.csv", header = T) 
head(SIFT1T)
library(e1071)
library(rpart)
attach(SIFT1T)
####SVM Training####
SIFT1T <- subset(SIFT1T, select=-X)
x <- subset(SIFT1T, select = -Categories)
x <- subset(x, select = -Num_Categories)
y <- subset(SIFT1T, select = Num_Categories)
svm_model <- svm(Categories ~ ., data=SIFT1T)
summary(svm_model)
svm_model1 <- svm(x,y)
summary(svm_model1)
pred <- predict(svm_model1,x)
summary(pred)
pred_Categories = vector()
for (i in 1:2000){
 if (pred[i] > 0.5){
   pred_Categories[i] = "Dog"
 }
 else{
   pred_Categories[i] = "Chicken"
 }
}  
class = vector()
for (i in 1:2000) {
  if (t(Categories)[i] == pred_Categories[i]){
    class[i] = 1
  }
  else{
    class[i] = 0
  }
}
summary(class)
svm_tune <- tune(svm, train.x=x, train.y=y, 
                 kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)*10^-4))

print(svm_tune)
####Tree Training####
#tree_model <- rpart(x,y)
#summary(tree_model)
#pred2 <- predict(tree_model, x)
#summary(pred2)
