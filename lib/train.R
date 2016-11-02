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
pca_x = pca$x[,1:5]
head(pca_x)

pr_var <- (pca$sdev)^2
prop_varex <- pr_var/sum(pr_var)
plot(prop_varex, xlim=c(1,50),xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

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

# best model 
p = cv_svmTune$best.parameters
best.svm = svm(pca_x,y,cost = p[,4], gamma = p[,2],degree = p[,1],coef0 = p[,3],type = "C-classification")
pred_svm <- predict(best.svm,pca_x)
accur = sum(pred_svm == as.vector(as.matrix(y)))/2000
accur