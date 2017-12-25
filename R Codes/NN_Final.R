#Load Libraries
library(caret)
library(neuralnet)
library(ROSE)
library(DMwR)
library(plyr)
library(ggplot2)
library(ROCR)


# Data Loading
df <- as.data.frame(readxl::read_xls("Data Set for Assignment II - 1000 Records 75 Attributes.xls"))
df$`Record Serial Number` <- NULL

# Exploratory Analysis

#Finding near zero and zero variance features and remove those variables
outcome <- df$OUTCOME
nzv<-nearZeroVar(df[,1:ncol(df)-1], saveMetrics = T)
df1<-df[,which(nzv$nzv==F)]



# Principal Component Analysis
pca1 <- prcomp(df1, scale. = T)
pr_var<- pca1$sdev^2
prop_varex <- pr_var/sum(pr_var)

#First 30 variables cover 95% of the variance
nsel <- 30
df1 <- as.data.frame(pca1$x[,1:nsel])


norm_max_min <- function(x)
{
    normalized <- (x-min(x))/(max(x)-min(x))
    return(normalized)
}

## Data Splitting
df1 <- cbind(df1, outcome)
set.seed(1234)

trainIndex <- createDataPartition(outcome, p=0.65, list = FALSE, times = 1)
train.df <- df1[trainIndex,]

valid.df <- df1[-trainIndex,]
validIndex <- createDataPartition(valid.df$outcome, p=0.5, list = FALSE, times = 1)

test.df <- valid.df[-validIndex,]
valid.df <- valid.df[validIndex,]

# Use ROSE to do class balancing
n <- names(train.df[,1:nsel])
f <- as.formula(paste("outcome ~", paste(n, collapse = " + ")))
rs <- ROSE(f, data=train.df, N=1200,p=0.5)
train.df <- rs$data


#Normalize the data
train.df[,1:nsel] <-as.data.frame(sapply(train.df[,1:nsel], norm_max_min))
valid.df[,1:nsel] <-as.data.frame(sapply(valid.df[,1:nsel], norm_max_min))
test.df[,1:nsel] <-as.data.frame(sapply(test.df[,1:nsel], norm_max_min))


#Cost and F1 Score Function
cost <- function(conf_mat)
{
    cost <- (conf_mat[1,2]*(-9000)) + (conf_mat[2,1]*(-1000)) + (conf_mat[2,2]*(9000))
    return(cost)
}

fScore <- function(cm){
    cm <- cm$table
    fval <- (2*cm[4])/((2*cm[4]) + cm[2]+cm[3])
    return(fval)
}

model_accuracy <- function(model, train, valid, nsel)
{
    #Predict for train data
    pr.nn <- compute(model,train[,1:nsel])
    pred_train <-ifelse(pr.nn$net.result>0.5,1,0)
    cm<-confusionMatrix(pred_train, train$outcome)
    print(cm$table)
    print(cost(cm$table))
    
    #F1 Score
    print(fScore(cm))
    
    #ROC Curve
    pred<-prediction(labels = train$outcome, predictions = pred_train)
    perf <- performance(pred, measure = "tpr", x.measure = "fpr")
    plot(perf, main = "ROC curve",col = "blue", lwd = 3)
    abline(a = 0, b = 1, lwd = 2, lty = 2)
    
    #AUC Score
    perf.auc <- performance(pred, measure = "auc")
    print(unlist(perf.auc@y.values))
    
    #Predict on validation data
    valid.nn<-compute(model, valid[,1:nsel])
    pred_val <- ifelse(valid.nn$net.result>0.5,1,0)
    cm<-confusionMatrix(pred_val, valid$outcome)
    print(cm$table)
    print(cost(cm$table))
}


#NN - Final Model
set.seed(12321)

train.df <- rbind(train.df, valid.df)

n_folds <- 3
folds_i <- sample(rep(1:n_folds, length.out = nrow(train.df)))
cv_tmp <- rep(1:n_folds)
for (k in 1:n_folds) {
    print(paste0("Fold ",k))
    test_i <- which(folds_i == k)
    train_df <- train.df[-test_i, ]
    test_df <- train.df[test_i, ]
    
    nn1 <- neuralnet(f,
                     data = train.df,
                     hidden = c(25,12,4), rep = 4, learningrate = 0.01, threshold = 0.02,
                     act.fct = "logistic",
                     linear.output = FALSE,
                     lifesign = "maximum") #Cost: -12000, 109,2,57,7
    
    
    pred <- compute(nn1,test_df[,1:nsel])
    pred_train <-ifelse(pred$net.result>0.5,1,0)
    cm<-confusionMatrix(pred_train, test_df$outcome)
    cv_tmp[k] <- cost(cm$table)
    print(paste0("Cost for fold ", k, ":", cv_tmp[k]))
}

print(paste0("Average Cost:",mean(cv_tmp)))

#Model Accuracy
model_accuracy(nn1, train.df, test.df, nsel = nsel)