# Machine Learning in R - Problem Set 1

# Group:
# Aaron Kleinbercher, Matrikelnummer: 461217
# Giulia Assmann, Matrikelnummer: 465236
# Philipp Borghard, Matrikelnummer: 464518

# Installing Required Packages
install.packages(c("GLMsData", "ggplot2", "class", "colorspace", "caret", "glmnet", "leaps", "Rcpp"))

# Loading Packages
library(GLMsData)
library(ggplot2)
library(class)
library(colorspace)
library(caret)
library(glmnet)
library(leaps)
library(RcppCCTZ)
library(plotly)
library(ISLR)
library(Metrics)
library(boot)

##### TASK 1 #####

# Loading Data
data(lungcap)
options(max.print=100000) #Showing all Data
show(lungcap) #Checking Data
data("lungcap")
lungcap$Htcm=lungcap$Ht*2.54 #transforming the height data from inches to cm.


# EXERCISE 1:

lung_model = lm(log(FEV) ~ Age + Htcm + Gender + Smoke, data=lungcap) #defining the model called "lung_model"
summary(lung_model) #Running the regression

# log(FEV) = beta_0 + beta_1*Age + beta_2*Htcm + beta_3*Gender + beta_4*Smoke + error term
# Fitted Model: log(FEV) = -1.943998 + 0.023387*Age + 0.016849*Htcm + 0.029319*Gender - 0.046067*Smoke
# With dummy variable Gender being 0 if person i is female and 1 if person i is male
# And dummy variable Age being 0 if person i does not smoke and 1 if person i smokes

levels(lungcap$Gender) <- c(0,1) # Transforming the qualitative variable gender into 0 and 1 instead "F" and "M" respectively.

# EXERCISE 2:

# plotting FEV against the other variables
plot_ly(lungcap, x = lungcap$Age, y = lungcap$FEV, type = "scatter", marker = list(size=6, color = 'rgb(255, 25, 25)', line = list(
  color = 'rgb(0, 0, 0)', width = 1)))%>%
  layout(title = "Age vs FEV",
         xaxis = list(title = "Age"),
         yaxis = list(title = "FEV"))

plot_ly(lungcap, x = lungcap$Htcm, y = lungcap$FEV, type = "scatter", marker = list(size=6, color = 'rgb(255, 25, 25)', line = list(
  color = 'rgb(0, 0, 0)', width = 1)))%>%
  layout(title = "Effect of Htcm on FEV",
         xaxis = list(title = "Htcm"),
         yaxis = list(title = "FEV"))

plot_ly(lungcap, x = lungcap$Gender, y = lungcap$FEV, type = "box", fillcolor = 'rgb(212, 21, 21)', line = list(
  color = 'rgb(0, 0, 0)'))%>%
  layout(title = "Gender vs FEV",
         xaxis = list(title = "Gender"),
         yaxis = list(title = "FEV"))

plot_ly(lungcap, x = lungcap$Smoke, y = lungcap$FEV, type = "box", fillcolor = 'rgb(212, 21, 21)', line = list(
  color = 'rgb(0, 0, 0)'), marker = list(size=6, color = 'rgb(212, 21, 21)', line = list(
    color = 'rgb(0, 0, 0)', width = 1)))%>%
  layout(title = "Effect of Smoking on FEV",
         xaxis = list(title = "Smoker"),
         yaxis = list(title = "FEV"))

# As you can see in the scatterplot the relationship between Htcm and FEV seems to be non-linear relationship.
# The relationship is rather quadratic than linear.

# Thats why we use log(FEV) as a response because it better fits the relationship:
# Plotting log(FEV) against Htcm

plot_ly(lungcap, x = lungcap$Htcm, y = log(lungcap$FEV), type = "scatter", marker = list(size=6, color = 'rgb(255, 25, 25)', line = list(
  color = 'rgb(0, 0, 0)', width = 1)))%>%
  layout(title = "Effect of Height on log(FEV)",
         xaxis = list(title = "Htcm"),
         yaxis = list(title = "log(FEV)")) 

# EXERCISE 3:

summary(lung_model)

# The Estimate column gives the estimated regression coefficients.
# The interpretation of the coefficient is that when all other variables are kept constant and the coefficient
# increases by one unit then on average the log(FEV) increases by the value of the coeffcient.

# For example, an increase in height of one cm is associated with an increase in the log(FEV) by 0.016849, when
# keeping all other variables constant. The other quantitative variable Age can be interpreted in the same way.
# Parameter estimates for qualitative variables indicate how much the value of explanatory variable changes.
# For example the value of log(FEV) will change by a factor of 0.046067 for smokers (Smoke=1),
# compared to non-smokers (Smoke=0). The other qualitative variable Gender can be interpreted in the same way.

# The standard errors of the regression coefficients assess the precision of each coefficient (how far the sample coefficient
# is likely to be from the population coefficent). The higher the standard error the worse the precision of the coefficient
# and thus the model predictions. Approximately 95% of repeated estimates should fall within plus/minus 2*std error, which is
# also a quick apprixmation of a 95% confidence intervall.
# Age: The true mean of the population age is with 95% probability within the interval of 0.023387 +/- 2*0.003348
# Gender: The true mean of the population gender coefficient is with 95% probability within the interval of 0.029319 +/- 2*0.011719

# Residual standard error: The residual standard error is the square root of the residual sum of squares
# divided by the residual degrees of freedom. The smaller the value the better is the fit of the regression line.
# The standard deviation of the residuals calculates how much the data points spread around the regression line.

# The F-statistic (overall model significance) is used to test for hypothesis that all regression coefficients are zero:
# H0: beta_1 = beta_2 = ... = beta_p = 0 vs H1: at least one beta is != 0
# If the p-value is less than 0.05, we reject the hypothesis on a 5% singificance level that there are
# no coefficients with effect on the outcome in the model.

# EXERCISE 4:

# The proportion of variability explained by the fitted lung_model is R-squared:
# The model can explain 81.06% of the variation.

# EXERCISE 5:

new <- data.frame(Age = 14, Htcm = 175, Gender = "M", Smoke = 0) # create new data frame
pred <- predict(lung_model, new) # predict log(FEV) for new data
pred
# The best guess for the log(FEV) is 1.361271.

logf.ci = predict(lung_model, new, level = 0.95, interval ="prediction") #creating 95% prediction interval
fev = exp(logf.ci) # inversing  log(FEV) to FEV
fev

# The interval [2.928889, 5.196153] is rather wide so it gives us limited information.

# EXERCISE 6:

lung_model2 = lm(log(FEV) ~ Age + Htcm + Gender + Smoke + Age*Smoke, data=lungcap) # define new multiple linear regression model
# with interaction term
summary(lung_model2)

# The coefficient of the interaction term is -0.0116659. We can interpret the interaction term as the change in FEV 
# for a unit increase in Age given the person is a smoker.
# The negative value indicates that the FEV of a aging person who smokes decreases. This means that the negative effect
# of smoking (based on previous regression without interaction term) overweights the positive effect of
# getting older (based on previous regression without interaction term) on FEV.
# The p-value of 0.16863 for the interaction term is not statistically significant.
# In addition, compared to the previos model (lung_model) the coefficient Smoke is not singificant
# anymore, while Age is still highly significant.
# However, our R-squared increases which indicates a higher explanatory power of the model.


####### TASK 2 #######

setwd("/Users/Giulia/Documents/Studium Master/Machine Learning/Problem Set 1")
t_raw = read.csv("tennisdata.csv")
tn = na.omit(data.frame(y=as.factor(t_raw$Result),
                        x1=t_raw$ACE.1-t_raw$UFE.1-t_raw$DBF.1,
                        x2=t_raw$ACE.2-t_raw$UFE.2-t_raw$DBF.2))
set.seed(4268) # for reproducibility
tr = sample.int(nrow(tn),nrow(tn)/2)
trte=rep(1,nrow(tn))
trte[tr]=0
tennis=data.frame(tn,"istest"=as.factor(trte)) # creating test data set
ggplot(data=tennis,aes(x=x1,y=x2,colour=y,group=istest,shape=istest))+
  geom_point()+theme_minimal()

# EXERCISE 1

# The dataframe consists of 787 observations.

median(tennis$x1) #median of all players labeled as "player 1" in the entire tournamend (50% better and 50% worse)
# median x1=-24
median(tennis$x2) #median of all players labeled as "player 2" in the entire tournamend (50% better and 50% worse)
# median x2=-24

# Plot Player Quality against Match Outcome
tn$y = factor(tn$y, levels = c(0, 1), labels = c("Player 2 Wins", "Player 1 Wins")) # Label Outcome (0 = Player 2 Wins, 1 = Player 1 Wins)

plot_ly(tn,x=tn$x1,y=tn$y,type="scatter", marker = list(size=6, color = 'rgb(250, 198, 010)', line = list(
  color = 'rgb(0, 0, 0)', width = 1))) %>%
  layout(title = "Match Outcome dependent on Quality of Player 1",
         xaxis = list(title = "Quality of Player 1"),
         yaxis = list(title = "Match Outcome"))

plot_ly(tn,x=tn$x2,y=tn$y,type="scatter", marker = list(size=6, color = 'rgb(250, 198, 010)', line = list(
  color = 'rgb(0, 0, 0)', width = 1))) %>%
  layout(title = "Match Outcome dependent on Quality of Player 2",
         xaxis = list(title = "Quality of Player 2"),
         yaxis = list(title = "Match Outcome"))


# If player 1 wins the match (y=1) its qualitiy tends to be higher (higher values of x1). For our scatterplot this means we don't
# find many observations in the top left (players with bad qualitiy usually don't win the match) and in the bottom right
# (good players usually don't lose the match).
# If player 2 wins the match (y=0) its qualitiy tends to be higher (higher values of x2). For our scatterplot this means we don't
# find many observations in the bottom left (players with bad qualitiy usually don't win the match) and in the top right
# (good players usually don't lose the match).

pairs(tn) #show pairwise scatterplots

# EXERCISE 2

tennis$y = factor(tennis$y, levels = c(0, 1), labels = c("Player 2 Wins", "Player 1 Wins")) # Label Match Outcome
tennis_model <- as.formula(y ~ x1 + x2) # define tennis model
logit <- glm(tennis_model, family = "binomial", data = tennis) # logistic regression with full data set
summary(logit) # print result

# Interpretation of results: p-values of both coefficients x1 and x2 indicate that they are significant on a
# 1% significance level. We have evidence to believe that the quality of the players influences the outcome of the match.

quality <- data.frame(x1 = -25, x2 = -20 ) # create data frame for observed qualities
pred_tennis <- predict(logit, quality, type = "response") # predict probabilty for y for given qualities.
pred_tennis

# The predicted probability that player 1 wins the match is 38.29%.
# Looking at our data tennis we see in the first row that the actual outcome was "0" meaning that player 2 won
# which is in line with our model (predicted probability being smaller than 50%)

# EXERCISE 3

quality_all <- data.frame(x1 = tennis$x1, x2 = tennis$x2)  # create data frame for all observed qualitys
pred2 <- predict(logit, quality_all, type = "response") # predict probabilty for y for all given qualities.

actual_tennis_all <- tennis$y # define the actual match outcomes

threshold <- 0.5

pred_tennis_all <- ifelse(pred2 > threshold, 1, 0) # if probability > threshold player 1 wins, else player 1 loses
pred_tennis_all = factor(pred_tennis_all, levels = c(0, 1), labels = c("Player 2 Wins", "Player 1 Wins")) # Label Match Outcome
confusionMatrix(pred_tennis_all, actual_tennis_all, positive = "Player 1 Wins") # calculate confusion matrix for trainig data

# Sensitivity: 0.7266
# Secificity: 0.7679

# As you can see from our confusion matrix the diagonale gives us the number of correct predictions
# (301 predicted that player 1 loses and he actually lost; 287 predicted that player 1 wins and he actually won).
# Also you see a acurancy of 0.7471 of the confusion matrix, which are our correct predictions (301+287) divided by
# the total number of observations (787).

# Null Rate:

table(tn$y)

# If we always predict player 1 wins we have an accuracy of 395/787 = 0.501906 (no information Rate).

# EXERCISE 4

test_tennis <- tn[-tr,]
training_tennis <- tn[tr, ]

logit2 <- glm(tennis_model, family="binomial", data = training_tennis) # logistic regression for training data only
summary(logit2) # print result

quality_test <- data.frame(x1 = test_tennis$x1, x2 = test_tennis$x2)  # create data frame for qualities for test data
pred3 <- predict(logit2, test_tennis, type = "response") # predict probabilty for y for all given qualities for test data

actual_tennis_test <- test_tennis$y # define the actual match outcomes for test data

pred_tennis_test <- ifelse(pred3 > threshold, 1, 0) # if probability > threshold player 1 wins, else player 1 loses
pred_tennis_test = factor(pred_tennis_test, levels = c(0, 1), labels = c("Player 2 Wins", "Player 1 Wins")) # Label Match Outcome
confusionMatrix(pred_tennis_test, actual_tennis_test, positive = "Player 1 Wins") # calculate confusion matrix for test data

# misclassification error: 1-accuracy --> 1-0.7665 = 0.2335

# EXERCISE 5

require(MASS)
discr <- lda(tennis_model, data = training_tennis) # define linear discriminant analysis with training data only
summary(discr)

pred4 <- predict(discr, test_tennis) # predict probabilty for y for all given qualities for test data

confusionMatrix(pred4$class, actual_tennis_test, positive = "Player 1 Wins") # create confusion matrix

# misclassification error: 1-accuracy --> 1-0.764 = 0.236

# EXERCISE 6

data_new <- pred4$posterior # define new data set with probabilities

count <- data_new[data_new >= 0.8] # create vector with probabilities larger than 0.8
length(count) # count vector entries

# The outcome of 106 matches can be predicted with a probability larger than 80%.

# EXERCISE 7

ks = 1:30
yhat = sapply(ks, function(k){
  class::knn(train=tn[tr,-1], cl=tn[tr,1], test=tn[,-1], k = k) # test both train and test
})
train.e = colMeans(tn[tr,1]!=yhat[tr,])
test.e = colMeans(tn[-tr,1]!=yhat[-tr,])

set.seed(0)
ks = 1:30 # Choose K from 1 to 30.
idx = createFolds(tn[tr,1], k=5) # Divide the training data into 5 folds.
# "Sapply" is a more efficient for-loop.
# We loop over each fold and each value in "ks"
# and compute error rates for each combination.
cv = sapply(ks, function(k){
  sapply(seq_along(idx), function(j) {
    yhat = class::knn(train=tn[tr[ -idx[[j]] ], -1],
                      cl=tn[tr[ -idx[[j]] ], 1],
                      test=tn[tr[ idx[[j]] ], -1], k = k)
    mean(tn[tr[ idx[[j]] ], 1] != yhat)
  }) })

# The entry (j,k) of the matrix cv is the CV error rate for test fold j=5 where K=30 for the KNN-classifier.

cv.e = colMeans(cv) # matrix cv
cv.se = apply(cv, 2, sd)/sqrt(5)
k.min = which.min(cv.e) # min CV error rate
print(k.min)

# Given our mean misclasification error our optimal K is K=25.

# EXERCISE 8

library(colorspace)
co = rainbow_hcl(3)
par(mar=c(4,4,1,1)+.1, mgp = c(3, 1, 0))
plot(ks, cv.e, type="o", pch = 16, ylim = c(0, 0.7), col = co[2],
     xlab = "Number of neighbors", ylab="Misclassification error")
arrows(ks, cv.e-cv.se, ks, cv.e+cv.se, angle=90, length=.03, code=3, col=co[2])
lines(ks, train.e, type="o", pch = 16, ylim = c(0.5, 0.7), col = co[3])
lines(ks, test.e, type="o", pch = 16, ylim = c(0.5, 0.7), col = co[1])
legend("topright", legend = c("Test", "5-fold CV", "Training"), lty = 1, col=co)

# The bias will increase with K, but the variance will decrease.
# With high flexibility (k=1) of the model the bias is very low but the variance on the test data is high.
# With growing number of k, the flexibiliby decreases and the bias increases but the variance on the test data decreases.

# EXERCISE 9

k = 30
size = 100
xnew = apply(tn[tr,-1], 2, function(X) seq(min(X), max(X), length.out=size))
grid = expand.grid(xnew[,1], xnew[,2])
grid.yhat = knn(tn[tr,-1], tn[tr,1], k=k, test=grid)
np = 300
par(mar=rep(2,4), mgp = c(1, 1, 0))
contour(xnew[,1], xnew[,2], z = matrix(grid.yhat, size), levels=.5,
        xlab=expression("x"[1]), ylab=expression("x"[2]), axes=FALSE,
        main = paste0(k,"-nearest neighbors"), cex=1.2, labels="")
points(grid, pch=".", cex=1, col=grid.yhat)
points(tn[1:np,-1], col=factor(tn[1:np,1]), pch = 1, lwd = 1.5)
legend("topleft", c("Player 1 wins", "Player 2 wins"),
       col=c("red", "black"), pch=1)
box()

# The graph shows the decision border of the nearest neighbor model for the prediction of the match outcome.
# The k is set to 30. So the decision of the model is based on the majority of the 30 nearest neighbors of an observation.

# EXERCISE 10

# K = 1 Nearest Neighbors
k = 1
size = 100
xnew = apply(tn[tr,-1], 2, function(X) seq(min(X), max(X), length.out=size))
grid = expand.grid(xnew[,1], xnew[,2])
grid.yhat = knn(tn[tr,-1], tn[tr,1], k=k, test=grid)
np = 300
par(mar=rep(2,4), mgp = c(1, 1, 0))
contour(xnew[,1], xnew[,2], z = matrix(grid.yhat, size), levels=.5,
        xlab=expression("x"[1]), ylab=expression("x"[2]), axes=FALSE,
        main = paste0(k,"-nearest neighbors"), cex=1.2, labels="")
points(grid, pch=".", cex=1, col=grid.yhat)
points(tn[1:np,-1], col=factor(tn[1:np,1]), pch = 1, lwd = 1.5)
legend("topleft", c("Player 1 wins", "Player 2 wins"),
       col=c("red", "black"), pch=1)
box()

# K = 50 Nearest Neighbors

k = 50
size = 100
xnew = apply(tn[tr,-1], 2, function(X) seq(min(X), max(X), length.out=size))
grid = expand.grid(xnew[,1], xnew[,2])
grid.yhat = knn(tn[tr,-1], tn[tr,1], k=k, test=grid)
np = 300
par(mar=rep(2,4), mgp = c(1, 1, 0))
contour(xnew[,1], xnew[,2], z = matrix(grid.yhat, size), levels=.5,
        xlab=expression("x"[1]), ylab=expression("x"[2]), axes=FALSE,
        main = paste0(k,"-nearest neighbors"), cex=1.2, labels="")
points(grid, pch=".", cex=1, col=grid.yhat)
points(tn[1:np,-1], col=factor(tn[1:np,1]), pch = 1, lwd = 1.5)
legend("topleft", c("Player 1 wins", "Player 2 wins"),
       col=c("red", "black"), pch=1)
box()

# K = 300 Nearest Neighbors

k = 300
size = 100
xnew = apply(tn[tr,-1], 2, function(X) seq(min(X), max(X), length.out=size))
grid = expand.grid(xnew[,1], xnew[,2])
grid.yhat = knn(tn[tr,-1], tn[tr,1], k=k, test=grid)
np = 300
par(mar=rep(2,4), mgp = c(1, 1, 0))
contour(xnew[,1], xnew[,2], z = matrix(grid.yhat, size), levels=.5,
        xlab=expression("x"[1]), ylab=expression("x"[2]), axes=FALSE,
        main = paste0(k,"-nearest neighbors"), cex=1.2, labels="")
points(grid, pch=".", cex=1, col=grid.yhat)
points(tn[1:np,-1], col=factor(tn[1:np,1]), pch = 1, lwd = 1.5)
legend("topleft", c("Player 1 wins", "Player 2 wins"),
       col=c("red", "black"), pch=1)
box()

# Comparing the graphs: the lower the choosen k the higher is the flexibility of the classification line. With k = 300 the line is
# closer to a linear decision boundary. The lower k the lower the bias (at k = 1 it is always zero for the training sample
# since the closest point to any training data is itself). However, a lower k results in overfitting and a high validation error.
# Our model does not generalize enough and thus has not much validity and a high variance.
# The larger k becomes the less flexible is the model but is expected to have a lower variance.

# K = 500 Nearest Neighbors

k = 500
size = 100
xnew = apply(tn[tr,-1], 2, function(X) seq(min(X), max(X), length.out=size))
grid = expand.grid(xnew[,1], xnew[,2])
grid.yhat = knn(tn[tr,-1], tn[tr,1], k=k, test=grid)
np = 300
par(mar=rep(2,4), mgp = c(1, 1, 0))
contour(xnew[,1], xnew[,2], z = matrix(grid.yhat, size), levels=.5,
        xlab=expression("x"[1]), ylab=expression("x"[2]), axes=FALSE,
        main = paste0(k,"-nearest neighbors"), cex=1.2, labels="")
points(grid, pch=".", cex=1, col=grid.yhat)
points(tn[1:np,-1], col=factor(tn[1:np,1]), pch = 1, lwd = 1.5)
legend("topleft", c("Player 1 wins", "Player 2 wins"),
       col=c("red", "black"), pch=1)
box()

# For k = 500 the model has no decision border anymore, because we have less data points than k neighbors.
# Thus the model always decides to go with the majority group (the null rate).


##### TASK 3 #####

data(Carseats)
options(max.print=100000) # Showing all Data
show(Carseats) # Checking Data

# EXERCISE 1

lm.fit_cs = glm(Sales ~ Price + Urban + US, data=Carseats) # defining the multiple linear regression model
summary(lm.fit_cs)

# EXERCISE 2 A

set.seed(2019) # for reproducibility
train_test = sample(1:nrow(Carseats), 200, replace = FALSE)
train_carseats = Carseats[train_test,] # define train set
test_carseats = Carseats [-train_test, ] # define test set

# EXERCISE 2 B

lm.fit_cstr = glm(Sales ~ Price + Urban + US, data=train_carseats) # defining the multiple linear regression model
summary(lm.fit_cstr)

# Comparison (first value is based on full data set and second on trainings data):
# Intercept: In both regressions positive and highly significant. Values are very similar. (13.043469, 13.359223)
# A rural (not urban) store and a abroad (not US) store generates sales of ~13 thousands for a price of zero.
# But we should be careful with the interpretation because a price of zero usually does not generate any sales at all.
# Price: In both regressions negative and highly significant. Values are very similar (-0.054459, -0.052857)
# A price increase of one unit (1$) decreases the sales (in thousands) by ~0.05 (~50$)
# Urban: In both regressions not significant. Values differ between the two regressions (-0.021916, -0.377402)
# US: In both regressions positive and highly significant. Values are similar. (1.200573, 1.102264).
# If we have a store in the US, sales (in thousands) increases by ~1.0-1.2.This means that only moving a store from
# abroad to the US increases the sales (in thousands) by ~1.0-1.2.

# For all coefficients the standard error increases if we only regress with the training data. This is reasonable
# since the number of observations decreases from 400 to 200. Thus also the p values increase.

# EXERCISE 2 C

new_carseats <- data.frame(Price = Carseats$Price, Urban = Carseats$Urban, US = Carseats$US) # create data frame with required information
pred5 <- predict(lm.fit_cstr, new_carseats) # predict the response for all 400 observations

c <- mse(Carseats$Sales, pred5) # calculate MSE
c
# The MSE is 6.116983.

# EXERCISE 2 D

# use set.seed(2018)

set.seed(2018)
train_test1 = sample(1:nrow(Carseats), 200, replace = FALSE) # for reproducibility
train_carseats1 = Carseats[train_test1,] # define train set
test_carseats1 = Carseats [-train_test1, ] # define test set

lm.fit_cstr1 = glm(Sales ~ Price + Urban + US, data=train_carseats1) # define multiple linear regression model
summary(lm.fit_cstr1)

pred6 <- predict(lm.fit_cstr1, new_carseats) # predict the response for all 400 observations
d1 <- mse(Carseats$Sales, pred6) # calculate mean squared error

# Use set.seed(2020)

set.seed(2020)
train_test2 = sample(1:nrow(Carseats), 200, replace = FALSE) # for reproducibility
train_carseats2 = Carseats[train_test2,] # define train set
test_carseats2 = Carseats [-train_test2, ] # define test set

lm.fit_cstr2 = glm(Sales ~ Price + Urban + US, data=train_carseats2) # define multiple linear regression model
summary(lm.fit_cstr2)

pred7 <- predict(lm.fit_cstr2, new_carseats) # predict the response for all 400 observations
d2 <- mse(Carseats$Sales, pred7) # calculate mean squared error

# Print MSE for set.seed(2018) and set.seed(2020)
d1
d2

# If we use set.seed(2018) the MSE decreases to 6.059012.
# If we use set.seed(2020) the MSE decreases to 6.073237

# EXERCISE 2 E

cv.error <- cv.glm(lm.fit_cstr, data = train_carseats) # Compute the LOOCV estimate for the MSE
cv.error$delta

# MSE: 6.439424
# MSE bias corrected: 6.438828

# Exercise 3

reg <- regsubsets(Sales ~ CompPrice + Income + Advertising + Population + Price + Age + Education + Urban + US, data = Carseats, nvmax = 3)
summary(reg)

# The chosen predictors indicate Price, CompPrice and Advertising as the 3 most important variables.

lm.fit_regsubset = glm(Sales ~ Price + CompPrice + Advertising, data=train_carseats) # define multiple regression model

cv.error1 <- cv.glm(lm.fit_regsubset, data=train_carseats) # Compute the LOOCV estimate for the MSE
cv.error1$delta

# The LOOCV estimate for the MSE decreases.
# MSE: 4.522707
# MSE biased corrected: 4.522237

# EXERCISE 4

mean(Carseats$Sales) # Mean of 7.496325.
set.seed(1)

boot_mean_results <- replicate(20, mean(sample(Carseats$Sales, 400, replace = TRUE )))
summary(boot_mean_results)
# min(boot_mean_results): 7.118
# max(boot_mean_results): 7.736

# EXERCISE 5

set.seed(1)

meanfun <- function (data, i){
  d<-data [i,]
  return (mean (d))
  }

mean_results <- boot(Carseats[, "Sales", drop = FALSE], statistic = meanfun , R = 20)
summary(mean_results)
min(mean_results$t)
# 7.22255
max(mean_results$t)
# 7.6662
# The mean for the bootstrapped samples lies between 7.292475 and 7.700225.

boot_ci = boot.ci(boot.out = mean_results, conf = 0.95, type = "norm")
boot_ci
# The 95% confidence interval is (7.341,  7.797).

##### TASK 4 #####

set.seed(1007) # for reproducibility
sm = sample.int(nrow(diamonds),nrow(diamonds)/10)
smbi=rep(1,nrow(diamonds))
smbi[sm]=0
diamonds2=data.frame(diamonds,"issmall"=as.factor(smbi))
small = (diamonds2$issmall==0)
diamondssm=diamonds2[small,]

# EXERCISE 1

highest_price <- diamondssm[order(diamondssm$price, decreasing=TRUE),] # sorting the data
highest_price[1:3, ] # determine three highest prices in the dataset

# The highest prices are 18795 (carat = 2.00), 18779 (carat = 2.06) and 18759 (carat = 2.00)

summary(diamondssm) # overview over data

# Mean of the weight is 0.8002 carat , most prevalent color is G.

# Plot Carat vs Price
plot_ly(diamondssm, x= diamondssm$carat, y=diamondssm$price, type="scatter", marker = list(symbol = "diamond-tall", size=6, color = 'rgb(52, 235, 204)', line = list(
  color = 'rgb(0, 0, 0)', width = 1))) %>%
  layout(title = "Carat vs Price",
         xaxis = list(title = "Carat"),
         yaxis = list(title = "Price"))
# Plot log(Carat) vs log(Price)

plot_ly(diamondssm, x= log(diamondssm$carat), y= log(diamondssm$price), type="scatter", marker = list(symbol = "diamond-tall", size=6, color = 'rgb(52, 235, 204)', line = list(
  color = 'rgb(0, 0, 0)', width = 1))) %>%
  layout(title = "log(Carat) vs log(Price)",
         xaxis = list(title = "log(Carat)"),
         yaxis = list(title = "log(Price)"))

# EXERCISE 2

diamondssm=data.frame(diamondssm,"logprice"=log(diamondssm$price),
                      "logcarat"=log(diamondssm$carat))
diamonds3 <- subset(diamondssm, select = -c(price, carat, issmall))

model_forward <- regsubsets(logprice ~ ., data = diamonds3, nvmax=23, method = "forward") # forward stepwise selection
model_backward <- regsubsets(logprice ~ ., data = diamonds3, nvmax=23, method = "backward") # backward stepwise selection

summary(model_forward) # Get an overview over coefficients 
summary(model_backward)  # Get an overview over coefficients

which.max(summary(model_forward)$adjr2) # Find number of coefficients of best subset from forward stepwise selection
plot(summary(model_forward)$adjr2, xlab = "Number of Variables", ylab = "Adjusted R^2", type = "b")

which.max(summary(model_backward)$adjr2) # Find number of coefficients of best subset from backward stepwise selection
plot(summary(model_backward)$adjr2, xlab = "Number of Variables", ylab = "Adjusted R^2", type = "b")

# Using adjusted R^2 as criterion the best subset from a backward and forward stepwise selection has 17 predictors.

coef(model_forward, 17) # Find coefficients from best subset 
coef(model_backward, 17) # Find coefficients from best subset

# Both methods yield the same results for the best subsets of predictors. 
# The model contains the coefficients: cut.L, cut.Q, cut.C, color.L, color.Q, color.C, color^4, clarity.L, clarity.Q, 
# clarity.C, clarity^4, clarity^5, clarity^7, depth, x, z, logcarat.

# EXERCISE 3

# Comparison between R-squared and CV:
# Main goal of both methods is to evaluate the true prediction error.
# Cross Validation: This procedure has an advantage relative to adjusted R2,in that it provides
# a direct estimate of the test error, and makes fewer assumptions about the true underlying model.
# It can also be used in a wider range of model selection tasks, even in cases where it is hard to pinpoint the model
# degrees of freedom (e.g. the number of predictors in the model) or hard to estimate the error variance.
# In the past performing Cross Validation was more complicated because computers did not have enough CPU power.
# But today it is a very attractive approach for selecting from among a number of models under consideration.
# Adjusted R-squared (indirect method): A large value of adjusted R2 indicates a model with a small test error based on the
# size of the training error.

# EXERCISE 4

# Data Preparation 
x_diamonds3 <- model.matrix(logprice ~ ., data = diamonds3)[,-1]
y_diamonds3 <- diamonds3$logprice

# Ridge Regression
set.seed(100) # For Reproducability 
ridge_model = cv.glmnet(x_diamonds3, y_diamonds3, lambda = NULL, alpha = 0, nfolds = 10) # ridge regression
ridge_model
plot(ridge_model)

# Find Minimum MSE Error and the corresponding lambda 
ridge_model$lambda.min # Find best lambda
min(ridge_model$cvm) # Find corresponding MSE

# Show values of the coefficients
coef(ridge_model)

# The optimal lambda for Ridge Regression is 0.09814251 with a MSE of 0.02747375.

# Lasso Regression
set.seed(100)
lasso_model <- cv.glmnet(x_diamonds3, y_diamonds3, alpha = 1, nfolds = 10) # lasso regression
lasso_model
plot(lasso_model)

# Find Minimum MSE Error and the corresponding lambda 
lasso_model$lambda.min # Find best lambda
min(lasso_model$cvm) # Find corresponding MSE

# Show values of the coefficients
coef(lasso_model)

# The optimal lambda for Lasso Regression is 0.001004519 with a MSE of 0.01783556.

# Multiple Linear Regression
set.seed(100)
train.control_LinModel= trainControl(method = "cv", number = 10) # To make linear model comparable to Ridge and Lasso regression
simple_lin_model = train(logprice ~., data = diamonds3, method = "lm", trControl = train.control_LinModel) # linear model
summary(simple_lin_model)
# Access MSE of linear model
(simple_lin_model$results$RMSE)^2

# The MSE of the multiple linear model is 0.01783799

# comparison of the three models:
# Ridge Model: 
# The ridge approach does not eliminate coefficients, but only decreases them (converge to zero). It seems that the coefficents
# color.C, color^5, color^6, cut^4, clarity^4, clarity^5 are not significant since they are very close to zero.
# The ridge model has 23 coefficients. 

# Lasso Model: 
# The lasso model eliminates the less important coefficients so that we only have 17 coefficients left. The lasso model has a lower
# MSE than the ridge model (MSE: 0.01783556 < 0.02747375).

# Multiple Linear Model 
# The multiple linear model with all coefficents has a slightly higher MSE than the lasso, but has a lower MSE than the ridge regression. 
# MSE: 0.01783556 (lasso) < 0.01783799 (linear model) < 0.02747375 (ridge)

# We wonder that the ridge model performs worse than the linear model. 
# When we set the lambda of the ridge regression to 0 then we receive better results then for the default lamdba sequence. . 