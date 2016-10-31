library(kernlab)
library(e1071)
library(rpart)
library(class)
data = read.csv(file.choose(), header = T)
dim(data)
head(data)[,1:6]
data$Num_Categories[1:1000] = 1
data$Num_Categories[1001:2000] = 0
data$Categories[data$Num_Categories==0] = "dog"
data$Categories[data$Num_Categories==1] = "chicken"
dim(data)
norm2 = data
head(norm2)[1:9]

attach(norm2)
####cleaning data####
norm2 <- subset(norm2, select=-c(X,X0))
x <- subset(norm2, select = -Categories)
x <- subset(x, select = -Num_Categories)
y <- subset(norm2, select = Num_Categories)


##### PCA #########
pca <- prcomp(x,scale = T)
head(pca$x)
plot(pca,type="lines")

# make prince compoent of 8 # 
pca_x = pca$x[,1:8]
head(pca_x)

pr_var <- (pca$sdev)^2
prop_varex <- pr_var/sum(pr_var)
plot(prop_varex, xlim=c(1,50),xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")


#### knn classification after PCA 
accuracy_knn = vector()
for (i in 1:50){
  knncv = knn.cv(pca_x, as.vector(as.matrix(y)), k = i)
  accuracy_knn[i] = sum(as.vector(as.matrix(y)) == knncv)/2000
}
knn_k = which.max(accuracy_knn)
knn_k
accuracy_knn[knn_k]
# final knn model 
knncv = knn(pca_x, as.vector(as.matrix(y)), k = knn_k)



### svm classification after PCA
# cross validation of 20 fold
tc <- tune.control(cross = 20)
Cs = c(.001,.01,.1,.5,1,10)
gammas = 10^(-3:2)
degres = c(0,1,2)
coef = c(0,1,2)
cv_svmTune <- tune.svm(pca_x, y =y, cost = Cs, gamma = gammas,degree = degres,coef0 = coef,
                       tunecontrol = tc)
summary(cv_svmTune)





################################### Reference #####################################
### linear SVM after PCA
svm_model_pca <- svm(y~pca_x)
summary(svm_model_pca)
pred_pca <- predict(svm_model_pca,pca_x)
summary(pred_pca)
pred_Categories_pca = vector()
for (i in 1:2000){
  if (pred_pca[i] > 0.5){
    pred_Categories_pca[i] = "chicken"
  }
  else{
    pred_Categories_pca[i] = "dog"
  }
}  
# accuracy 
sum(Categories==pred_Categories_pca)/length(pred_Categories_pca)



### non linear SVM after PCA
# test how many princepal component would be used for the non linear SVM
accuracy_svm = vector()
# SVM after pac
svm_model_pca2 <- svm(pca_x,y,kernal = sigmoid)
summary(svm_model_pca2)
pred_pca2 <- predict(svm_model_pca2,pca_x)
summary(pred_pca2)
pred_Categories_pca2 = vector()
for (i in 1:2000){
  if (pred_pca2[i] > 0.5){
    pred_Categories_pca2[i] = "dog"
  }
  else{
    pred_Categories_pca2[i] = "chicken"
  }
}
accuracy_svm= sum(Categories==pred_Categories_pca2)/2000








