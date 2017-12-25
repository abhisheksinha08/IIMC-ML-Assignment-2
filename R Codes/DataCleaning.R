#Load Libraries
library(caret)
library(neuralnet)
library(ROSE)
library(DMwR)
library(plyr)
library(ggplot2)


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

#Save to csv files
write.csv(train.df, file="train_df.csv", row.names = F)
write.csv(valid.df, file="valid_df.csv", row.names = F)
write.csv(test.df, file="test_df.csv", row.names = F)

rm(list = ls())
