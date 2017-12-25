#Load Libraries
library(caret)
library(rpart)
library(rpart.plot)
library(plyr)
library(ggplot2)
library(ROCR)
library(C50)

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

model_accuracy_dt <- function(model, train, valid, nsel)
{
    #Predict for train data
    pred_train <- predict(model,train[,1:nsel], type="class")
    cm<-confusionMatrix(pred_train, train$outcome)
    print(cm$table)
    print(cost(cm$table))
    
    #F1 Score
    print(fScore(cm))
    
    #ROC Curve
    pred <- ROCR::prediction(labels = train$outcome, predictions = as.integer(as.character(pred_train)))
    perf <- performance(pred, measure = "tpr", x.measure = "fpr")
    plot(perf, main = "ROC curve",col = "blue", lwd = 3)
    abline(a = 0, b = 1, lwd = 2, lty = 2)
    
    #AUC Score
    perf.auc <- performance(pred, measure = "auc")
    print(unlist(perf.auc@y.values))
    
    #Predict on validation data
    pred_val <- predict(model, valid[,1:nsel], type="class")
    #pred_val <- ifelse(valid.dt>0.5,1,0)
    cm<-confusionMatrix(pred_val, valid$outcome)
    print(cm$table)
    print(cost(cm$table))
}


#Load Datasets
train.df <- read.csv(file="train_df.csv")
valid.df <- read.csv(file="valid_df.csv")
test.df <- read.csv(file="test_df.csv")


nsel <- 25
n <- names(train.df[,1:nsel])
f <- as.formula(paste("outcome ~", paste(n, collapse = " + ")))

train.df<-rbind(train.df, valid.df)

set.seed(12321)
dt1 <-rpart(f, train.df, method = "class", control = rpart.control(cp=0.001,maxdepth = 25, xval=10))
model_accuracy_dt(dt1, train.df, valid.df, nsel = nsel)
#Cost: 14000, F1 Score: 0.85, AUC: 0.85

prp(dt1)

set.seed(12321)
cmod1 <- C5.0(train.df[,1:nsel], as.factor(train.df[,'outcome']), control = C5.0Control(earlyStopping = T))
    #rpart(f, train.df, method = "class", control = rpart.control(cp=0.0001,maxdepth = 10))
model_accuracy_dt(cmod1, train.df, valid.df, nsel = nsel)
#Cost: -17000, F1 Score: 0.91, AUC: 0.91

