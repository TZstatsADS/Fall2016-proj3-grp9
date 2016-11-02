library(kernlab)
library(e1071)
library(rpart)
library(class)
library(gbm)
data = read.csv(file.choose(), header = T)
dim(data)
head(data)[,1:6]
data$Num_Categories[1:1000] = 1
data$Num_Categories[1001:2000] = 0
data$Categories[data$Num_Categories==0] = "dog"
data$Categories[data$Num_Categories==1] = "chicken"
dim(data)
norm2_test = data
head(norm2_test)[1:9]

attach(norm2_test)
####cleaning data####
norm2_test <- subset(norm2_test, select=-c(X,X0))
x_test <- subset(norm2_test, select = -Categories)
x_test <- subset(x_test, select = -Num_Categories)
y_test <- subset(norm2_test, select = Num_Categories)


##### PCA #########
pca_test <- prcomp(x_test,scale = T)
head(pca_test$x)
plot(pca_test,type="lines")

pr_var <- (pca_test$sdev)^2
prop_varex <- pr_var/sum(pr_var)
plot(prop_varex, xlim=c(1,50),xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

# select the number of PC
pca_xtest = pca_test$x[,1:5]
head(pca_xtest)



######Testing Model Performance#####
pred_test <- predict(best.svm,pca_xtest)
accur_test = sum(pred_test == as.vector(as.matrix(y_test)))/2000
accur_test
