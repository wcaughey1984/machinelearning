#####################################################################################
# Title:	ML Final Project														#
# Purpose:	(Coursera) Code for Final Project in Machine Learning course of data	#
#			science specialization.													#
# Author:	Billy Caughey															#
# Date:		2018.03.12 - Initial Build												#
#           2018.03.13 - Still more building                                        #
#####################################################################################

##### Preprocessing #####
trainData <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
saveRDS(trainData,"mlfinal_trainingdata.rds")

## Test how many training observations I have
nrow(trainData)

## how many "classe"'s are there
table(trainData$classe)

# with 19,622 observations, it makes sense to develop a validation set
# It might be worth testing if a 60/40 split is good enough 

## Set up training and validation sets
set.seed(1234)
library(caret)
inTrain <- createDataPartition(trainData$classe, p = 0.7, list = FALSE)
t.data <- trainData[inTrain,]
v.data <- trainData[-inTrain,]

## Determine how much missing information there is...
library(Amelia)
missmap(t.data)	# everything after classe is essentially missing. 
                # There isn't enough information there to impute

t.data <- t.data[lapply(t.data, function(x) sum(is.na(x))/length(x)) < 0.1]
missmap(t.data)

# drop fields which can't be used in model
dropFields <- grep("timestamp|X|user_name|window|skewness|kurtosis|min|max|amplitude", 
                   names(t.data))
t.data <- t.data[,-dropFields]
names(t.data)

## Now, fit a tree with all fields 
library(caret)
#library(randomForest)
set.seed(1234)
train_control <- trainControl(method="cv",number=5)
mod1 <- train(classe ~ . , data = t.data, trControl = train_control, method = "rf")
print(mod1)
plot(mod1$finalModel)

##### Validataion Step #####
## Now, use the validation data to Validate the data
v.data <- v.data[,lapply(v.data, function(x) sum(is.na(x))/length(x)) < 0.1]
dropFields <- grep("timestamp|X|user_name|window|skewness|kurtosis|min|max|amplitude", 
                   names(v.data))
v.data <- v.data[,-dropFields]

valid.model <- predict(mod1,v.data)
 
table(valid.model,v.data$classe)
mean.class.error <- mean((valid.model != v.data$classe)); print(mean.class.error)
accuracy <- 1 - mean((valid.model != v.data$classe)); print(accuracy)

##### Testing Step #####
testData <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
saveRDS(testData,"mlfinal_testingdata.rds")

## Need to clean the test data the same way training and validation were cleaned
testData <- testData[,lapply(testData, function(x) sum(is.na(x))/length(x)) < 0.1]
dropFields <- grep("timestamp|X|user_name|window|skewness|kurtosis|min|max|amplitude", 
                   names(testData))
testData <- testData[,-dropFields]

test.predict <- predict(mod1,testData)
test.predict
