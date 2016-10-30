SIFT1 <- read.csv(file = "C:/Users/liran/OneDrive/TC Courses/ADS/Proj 3/Project3_poodleKFC_train/sift_features.csv", header = T) 
SIFT1T <- t(SIFT1)
SIFT1T <- read.csv(file = "C:/Users/liran/OneDrive/TC Courses/ADS/Proj 3/SIFT1T.csv", header = T) 
head(SIFT1T)
library(e1071)
library(rpart)
attach(SIFT1T)
library(kernlab)
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

#svm_tune <- tune(svm, train.x=x, train.y=y, 
                # kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)*10^-4))

#print(svm_tune)

###### non-linear SVM Training#####
#svp <- ksvm(x=x,y=as.vector(y),type="C-svc",kernel="rbf",kpar=list(sigma=1),C=1) 

svm_model2 <- svm(x,y, kernal = polinomial)
summary(svm_model2)
pred2 <- predict(svm_model2,x)
summary(pred2)
pred_Categories2 = vector()
for (i in 1:2000){
  if (pred2[i] > 0.5){
    pred_Categories2[i] = "Dog"
  }
  else{
    pred_Categories2[i] = "Chicken"
  }
}  
class2 = vector()
for (i in 1:2000) {
  if (t(Categories)[i] == pred_Categories2[i]){
    class2[i] = 1
  }
  else{
    class2[i] = 0
  }
}

summary(class2)


####Tree Training####
#tree_model <- rpart(x,y)
#summary(tree_model)
#pred2 <- predict(tree_model, x)
#summary(pred2)


###### SVM intergrated with PCA####
##### PCA #########
pca <- prcomp(x,scale = T)
summary(pca) # print variance accounted for 
loadings(pca) # pc loadings
pca$x
plot(pca,type="lines")

pca_x = pca$x[,1:70]
head(pca_x)

### linear SVM after PCA
svm_model_pca <- svm(pca_x,y)
summary(svm_model_pca)
pred_pca <- predict(svm_model_pca,pca_x)
summary(pred_pca)
pred_Categories_pca = vector()
for (i in 1:2000){
  if (pred_pca[i] > 0.5){
    pred_Categories_pca[i] = "Dog"
  }
  else{
    pred_Categories_pca[i] = "Chicken"
  }
}  
# accuracy 
sum(Categories==pred_Categories_pca)/length(pred_Categories_pca)

##### non linear SVM after PCA
# test how many princepal component would be used for the non linear SVM
accuracy_svm = vector()
for (j in 5:100){
  pca_x = pca$x[,1:j]
  # SVM after pac
  svm_model_pca2 <- svm(pca_x,y,kernal = sigmoid)
  summary(svm_model_pca2)
  pred_pca2 <- predict(svm_model_pca2,pca_x)
  summary(pred_pca2)
  pred_Categories_pca2 = vector()
  for (i in 1:2000){
    if (pred_pca2[i] > 0.5){
      pred_Categories_pca2[i] = "Dog"
    }
    else{
      pred_Categories_pca2[i] = "Chicken"
    }
  }  
  accuracy_svm[j-4] = sum(Categories==pred_Categories_pca2)/2000
}
accuracy_svm
plot(accuracy_svm~c(5:100),type = "b")



#### knn classification after PCA 
library(class)
accuracy_knn = vector()
for (i in 1:50){
  knncv = knn.cv(pca_x, as.vector(as.matrix(y)), k = i)
  accuracy_knn[i] = sum(as.vector(as.matrix(y)) == knncv)/2000
}
k = which.max(accuracy_knn)
knncv = knn.cv(pca_x, as.vector(as.matrix(y)), k = k)
sum(as.vector(as.matrix(y)) == knncv)/2000
