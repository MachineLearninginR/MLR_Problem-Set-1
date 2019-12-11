# Machine Learning in R - Problem Set 2

# Group:
# Aaron Kleinbercher, Matrikelnummer: 461217
# Giulia Assmann, Matrikelnummer: 465236
# Philipp Borghard, Matrikelnummer: 464518
# Christina Brinkmann, Matrikelnummer: 409901

# Installing Required Packages
install.packages(c("ISLR", "ggplot2", "randomForest", "gbm", "tree", "e1071", "mlr", "dendextend"))

# Loading Packages
library(ISLR)
library(ggplot2)
library(randomForest)
library(gbm)
library(tree)
library(e1071)
library(mlr)
library(dendextend)


##### TASK 1 #####

# EXERCISE 1
# Loading Data
data(Carseats)
set.seed(815) # for replication
train <- sample(1:nrow(Carseats), 200, replace = FALSE) # create a vector with 200 observations randomly picked from Carseats dataset
train_carseats <- Carseats[train,] # define train set
test_carseats <- Carseats [-train, ] # define test set 

# EXERCISE 2
set.seed(815) # for replication
tree <- tree(Sales ~ . - Sales, train_carseats) # define our tree called "tree" based on training data
summary(tree) # Get the used variables 
plot(tree) # plotting the regression tree
text(tree, pretty = 0, cex=0.5)

# Interpretation: The summary indicates that only six of the variables ("ShelveLoc", "Price", "Age", "Income", "CompPrice"
# Advertising") have been used for construction of the tree. The tree has 18 terminal nodes. It seems that the variable 
# "ShelveLoc" (i.e. the quality of the shelving location for the carseats) is the most important variable since it is
# the first splitting criterion. It splits the good shelving locations from bad and medium locations.
# The sum of squared errors for the tree is the residual mean deviance (2.147). 

tree.pred <- predict(tree, test_carseats)
mean((test_carseats$Sales - tree.pred)^2)
# The MSE (test error rate) for our regression tree is 4.518836.

# EXERCISE 3
set.seed(815)
cv_tree <- cv.tree(tree)
plot(cv_tree$size, cv_tree$dev, type = "b")
# Cross validation 
min_tree <- which.min(cv_tree$dev)
cv_tree$size[min_tree]
# We see the tree of size 18 does have the lowest deviation.
# Therefore we look for the second best model.
size_vs_dev <- data.frame(cv_tree$dev, cv_tree$size)
View(size_vs_dev)
# We prune to a size of 10 as the tree size seems to perform similar. 

prune_tree <- prune.tree(tree, best = 10)
plot(prune_tree)
text(prune_tree, pretty = 0)
prune.pred <- predict(prune_tree, test_carseats)
mean((test_carseats$Sales - prune.pred)^2)
# The MSE (test error rate) for our pruned regression tree increases to 4.590036.
# As expected the model performs a litte bit worse than the prior model with 18 terminal nodes. 

# EXERCISE 4
lrn.carseats <- makeLearner("regr.rpart")
traintask.carseats <- makeRegrTask(data = train_carseats, target = "Sales") 
set.seed(111)
resample.res <- resample(lrn.carseats, traintask.carseats, resampling = cv10, measures = mse, show.info = FALSE)
resample.res$measures.test

# The values show that the MSE between the 10 cross validation folds differ quite strongly. The reason for that could be 
# that we only have 200 observations. Also, we suppose that we have outliers in our dataset. 
# Dependent on the allocation of the outliers to the test or to the training dataset the model results (MSE) differ a lot. 


# EXERCISE 5
set.seed(815)
bag_tree <- randomForest(Sales ~ . , data = train_carseats, mtry=10, importance=TRUE)
bag_tree
bag.pred <- predict(bag_tree, test_carseats)

mean((test_carseats$Sales - bag.pred)^2)
# The MSE (test error rate) for the regression tree performed with bagging decreases to 2.718382.

importance(bag_tree)
# The three most important variables are ShelveLoc, Price and CompPrice. 
# In the case of omitting these variables the MSE would increase most (%IncMSE). 

# EXERCISE 6

# Growing a random forest proceeds in the same way as the bagging approach, except that we use smaller values for the mtry argument. 
# The mytry argument defines the number of predictors that are available for biulding the regression trees. 
# By default, randomForest() uses p/3 variables when building a random forest of regression trees. To see how mtry
# effects the error rate we run the random forest with different mtry and calculate the MSE on the test data. 

mse_rF = double(10)

set.seed(815)
for(i in 1:10 ){
rf_tree <- randomForest(Sales ~ . , data = train_carseats, mtry = i,  importance = TRUE)
rf.pred <- predict(rf_tree, newdata = test_carseats)
mse_rF[i] <- mean((test_carseats$Sales - rf.pred)^2)
}

which.min(mse_rF) # size of tree with minimial MSE
mse_rF[which.min(mse_rF)] # minimal MSE

# After making the random forest we see that the MSE is lowest for mtry = 5 (MSE: 2.635781)
# We observe that for small values of mtry the MSE is high. But when we allow for more predictors then the MSE decreases. 
# For more than 5 predictors the MSE increases again due to overfitting. 

set.seed(815)
rf.tree <- randomForest(Sales ~ . , data = train_carseats, mtry = 5,  importance = TRUE)
importance(rf.tree)

# We see that our 5 most imporatant variables are: ShelveLoc, Price, CompPrice, Age and Advertising. 
# For example the MSE increases by ~53% if we remove ShelveLoc from our model. 

# EXRECISE 7
set.seed(815)
boost_tree <- gbm(Sales ~ . , data  = train_carseats, distribution="gaussian" , n.trees = 100)
summary(boost_tree)

boost.pred <- predict(boost_tree, newdata = test_carseats,n.trees=100)
mean((boost.pred - test_carseats$Sales)^2)

# This boosted tree performs much better with a MSE of 1.918701 compared to the random forest with a MSE of 2.635781
# and compared to the bagging approach with  a MSE of 2.718382.

##### TASK 2 #####

load(url("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/ESL.mixture.rda")) 
# prob gives probabilites for each class when the true density functions are known #px1 and px2 are coordinates in x1
# (length 69) and x2 (length 99) where class probabilites are calculated
rm(x,y)
attach(ESL.mixture)
dat=data.frame(y=factor(y),x)
xgrid=expand.grid(X1=px1,X2=px2)
par(pty="s")
plot(xgrid, pch=20,cex=.2)
points(x,col=y+1,pch=20)
contour(px1,px2,matrix(prob,69,99),level=0.5,add=TRUE,col="blue",lwd=2) #optimal boundary

# EXERCISE 1

# Bayes classifier:  
# Given the feature values, we classify the prediction to the class where it has the highest conditional probability to be 
# classified in. To calculate the conditional probability P(Y|X) we need the prior probabilities P(X) and P(Y) and the 
# conditional probabilities P(X|Y). Applying Bayes theorem we can calculate the probabilities for classifying to Y1 or
# Y2 given the features X and decide based on the calculated probability P(Y1|X) > P(Y2|X).

# Bayes decision boundary: Is the boundary where the classification probability is the same for all classes. 
# In the two class case, the decision boundary represents the 50% probability of classifying to Y1 or Y2. 

# Bayes error rate: Is the lowest possible test error rate. It is also called irreducable error. 

# In the graph we try to classify observations by two variables. The blue line is called decision boundary and
# shows where the conditional probability for classification is 50%. 
# The observations above the blue line are classified red, because their conditional probability is above 50%. 
# The observations below the blue line are classified black, because their conditional probability is below 50%.

# EXERCISE 2

library(e1071)
# support vector classifier
svcfits=tune(svm,factor(y)~.,data=dat,scale=FALSE,kernel="linear",ranges=list(cost=c(1e-2,1e-1,1,5,10)))
summary(svcfits)
svcfit=svm(factor(y)~.,data=dat,scale=FALSE,kernel="linear",cost=0.01)
# support vector machine with radial kernel
set.seed(4268)
svmfits=tune(svm,factor(y)~.,data=dat,scale=FALSE,kernel="radial",ranges=list(cost=c(1e-2,1e-1,1,5,10),gamma=c(0.01,1,5,10)))
summary(svmfits)
svmfit=svm(factor(y)~.,data=dat,scale=FALSE,kernel="radial",cost=1,gamma=5)
# the same as in a - the Bayes boundary
par(pty="s")
plot(xgrid, pch=20,cex=.2)
points(x,col=y+1,pch=20)
contour(px1,px2,matrix(prob,69,99),level=0.5,add=TRUE,col="blue",lwd=2) #optimal boundary
# decision boundaries from svc and svm added
svcfunc=predict(svcfit,xgrid,decision.values=TRUE)
svcfunc=attributes(svcfunc)$decision
contour(px1,px2,matrix(svcfunc,69,99),level=0,add=TRUE,col="red") #svc boundary
svmfunc=predict(svmfit,xgrid,decision.values=TRUE)
svmfunc=attributes(svmfunc)$decision
contour(px1,px2,matrix(svmfunc,69,99),level=0,add=TRUE,col="orange") #svm boundary

# EXERCISE 2 

# In the case that we know the decision boundary of the bayes classifier, we  would not need a test set anymore.
# The bayes error corresponds to the lowest possible test error / irreducable error. 
# The bayes classifier gives us the best test error rate and thus we would compare the bayes decision boundary 
# to the decision boundaries of the other trained models and evaluate the performance of the other models based 
# on the comparsion of the bayes decision boundary. 
# In this case we would not need a test dataset to generate another performance measure to compare the models.

# EXERCISE 3

# Support Vector Classifier have linear decision boundaries. When data is not linear seperable we need another approach. 
# Support Vector Machines is one. A SVM uses the kernel trick to make the problem linear seperable. Or we can use polynomials. 
# Thus at the end it generates non linear decision boundaries. 

# EXERCISE 4

# The parameter of a Support Vector Classifier is the C. C is the maximum proportion of misclassification we allow for the support vector. 
# If c is = 0 then we allow for no misclassification and we receive the output of a maximum margin classifier. 
# But when the observations are not perfectly linear seperable we need to allow for some misclassification by increasing C.  

# Additional to the parameter C, we can adjust the kernel of a support vector machine. 
# By applying a kernel function we add dimensions to the feature space and make the classification linear seperable. 

# EXERCISE 5

# The SVM compared to the Bayes decision boundary performs very good. In the region with many data points (x1:0-2; x2: 0-2) both decision 
# boundaries are very similiar. Nevertheless in the regions with less observations the decision boundaries differ. Specially in the 
# region x1: (-3)-(-1) and x2: (-2)-0. Also in the region of x1: 3-4 and x2: (-2)-0 the decision booundaries differ. There is one 
# outlier observation that seems to influence the SVM a lot. All in all both decision boundaries are similiar, but in regions with
# few observations the decision boundaries differ. 
# As a concrete measure of the performance of a model the squared distance between the decision boundaries could be calculated. 
# The model with the least deviation from the Bayes decision boundary would perform best.


##### TASK 3 #####

data("USArrests")
show(USArrests)

# EXERCISE 1
# Clustering with euclidean distance
USA.dist <- dist(USArrests, method = "euclidean")

# Hierarchical clustering using Complete Linkage
cluster.complete <- hclust(USA.dist, method = "complete" )

# Plot the obtained dendrogram
plot(cluster.complete, main = "Complete Linkage Clustering")

# EXERCISE 2
# Mathematical Approach 
USA.clusters <- cutree(cluster.complete, 5)
USA.clusters
table(USA.clusters)

# Graphical Approach 
plotteddendro = as.dendrogram(cluster.complete)
coloureddendro = color_branches(plotteddendro, k = 5)
plot(coloureddendro)
# Cluster 4 (pink) is the smallest with two states. It contains Florida and North Carolina.

# EXERCISE 3
USA.new <- scale(USArrests)
# Clustering with euclidean distance
USA.new.dist <- dist(USA.new, method = "euclidean")

# Hierarchical clustering using Complete Linkage
cluster.scaled.complete <- hclust(USA.new.dist, method = "complete" )

# Plot the obtained dendrogram
plot(cluster.scaled.complete)

# Redo EXERCISE 2:
USA.clusters.scaled <- cutree(cluster.scaled.complete, 5)
USA.clusters.scaled
table(USA.clusters.scaled)

# Scaling affects the clusters. 
# The scaling makes the variables comparable by accounting for different means and standard deviations. 
# In our case the variable "Assault" ranges from 45 to 337 whereas the variable "Murder" ranges only from 0.8 to 17.4. 
# Differences in the variable "Assault" influences the absolute distance more than differences in the variable "Murder". 
# By scaling all variables become equally important when calculating the distance.
# Cluster 2 is now the smallest cluster, only containing Alaska.

# EXERCISE 4
distance <- dist(USA.new)
distance
min(distance)
# The smallest distance is between New Hampshire and Iowa. The euclidean distance between these two states is 0.2058539.