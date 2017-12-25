#Load Libraries
library(caret)
library(neuralnet)
library(plyr)
library(ggplot2)
library(ROCR)

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


#Load Datasets
train.df <- read.csv(file="train_df.csv")
valid.df <- read.csv(file="valid_df.csv")
test.df <- read.csv(file="test_df.csv")

nsel <- 30
n <- names(train.df[,1:nsel])
f <- as.formula(paste("outcome ~", paste(n, collapse = " + ")))


#---------------------------------------
#Neural Networks
#---------------------------------------


#NN - 1
set.seed(12321)

nn1 <- neuralnet(f,
                 data = train.df,
                 hidden = c(8,3), rep = 5, learningrate = 0.01, threshold = 0.02,
                 act.fct = "logistic", 
                 linear.output = FALSE, 
                 lifesign = "maximum")
model_accuracy(nn1, train.df, valid.df, nsel = nsel)
##Cost:-31000, AUC:0.95, F1:0.94

#NN - 2
set.seed(12321)
nn2 <- neuralnet(f, data = train.df, hidden = c(6,2), rep = 10, learningrate = 0.01, threshold = 0.02,
                 act.fct = "logistic", linear.output = FALSE, lifesign = "maximum")
model_accuracy(nn2, train.df, valid.df, nsel = nsel)
##Cost:-95000, AUC:0.94, F1:0.939

#NN - 3
set.seed(12321)
nn3 <- neuralnet(f, data = train.df, hidden = c(10,3), rep = 10, learningrate = 0.01, threshold = 0.03,
                 act.fct = "logistic", linear.output = FALSE, lifesign = "maximum")
model_accuracy(nn3, train.df, valid.df, nsel = nsel)
##Cost:-48000, AUC:0.937, F1:0.936


# Combination of nn1, nn2, nn3
valid.nn<-compute(nn1, valid.df[,1:nsel])
pred_val1 <- ifelse(valid.nn$net.result>0.5,1,0)

valid.nn<-compute(nn2, valid.df[,1:nsel])
pred_val2 <- ifelse(valid.nn$net.result>0.5,1,0)

valid.nn<-compute(nn3, valid.df[,1:nsel])
pred_val3 <- ifelse(valid.nn$net.result>0.5,1,0)

pred_val <- ifelse(((pred_val1 + pred_val2 + pred_val3)/3)>0.5,1,0)
cm<-confusionMatrix(pred_val, valid.df$outcome)
print(cm$table)
print(cost(cm$table))

## Cost: -53000



#NN - 4
set.seed(12321)
nn4 <- neuralnet(f,
                 data = train.df,
                 hidden = c(7,3), rep = 5, learningrate = 0.01, threshold = 0.02,
                 act.fct = "logistic", 
                 linear.output = FALSE, 
                 lifesign = "maximum")
model_accuracy(nn4, train.df, valid.df, nsel = nsel)
##Cost:-62000, AUC:0.938, F1:0.937


#NN - 5
set.seed(12321)
nn4 <- neuralnet(f,
                 data = train.df,
                 hidden = c(15,5), rep = 10, learningrate = 0.01, threshold = 0.02,
                 act.fct = "logistic", 
                 linear.output = FALSE, 
                 lifesign = "maximum")
model_accuracy(nn4, train.df, valid.df, nsel = nsel)
##Cost:-77000, AUC:0.967, F1:0.967



#NN - 6
set.seed(12321)


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
                     hidden = c(25,12,2), rep = 4, learningrate = 0.01, threshold = 0.02,
                     act.fct = "logistic",
                     linear.output = FALSE,
                     lifesign = "maximum") #Cost: -14000, 89,1,77,8

    
    pred <- compute(nn1,test_df[,1:nsel])
    pred_train <-ifelse(pred$net.result>0.5,1,0)
    cm<-confusionMatrix(pred_train, test_df$outcome)
    cv_tmp[k] <- cost(cm$table)
    print(paste0("Cost for fold ", k, ":", cv_tmp[k]))
}
cv <- mean(cv_tmp)
cv

model_accuracy(nn1, train.df, valid.df, nsel = nsel)



#NN - 7
set.seed(12321)


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
cv <- mean(cv_tmp)
cv

model_accuracy(nn1, train.df, valid.df, nsel = nsel)



#NN - Final Model
set.seed(12321)

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
cv <- mean(cv_tmp)
cv

model_accuracy(nn1, train.df, valid.df, nsel = nsel)

