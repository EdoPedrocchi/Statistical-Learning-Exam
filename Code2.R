rm(list = ls())

   
library(readxl)  # For reading Excel files
library(ggplot2) # For data visualization
library(dplyr)   # For data manipulation
library(tidyr)   # For reshaping data



library(caret)
library(data.table)


dataset_path <- "/Users/pedrocchiedoardo/Desktop/esame statistical learning/Dataset2_Companies.xlsx"


data <- read_excel(dataset_path)
data<-as.data.frame(data)


########################## EDA #####################################################


print(head(data))

data$ID <- NULL

print(colnames(data))
print(head(data))
print(sum(is.na(data))) 
summary(data)
attach(data)



## Get default rate
def_perc<-sum(data$Default)/length(data$Default)
print(def_perc)




cor(data[, sapply(data, is.numeric)])

correlation_matrix <- cor(data[, sapply(data, is.numeric)])
print(correlation_matrix)

# Heatmap  correlation matrix
heatmap(correlation_matrix, main="Matrice di correlazione", col=heat.colors(10))



#  comparing the distribution of a variable between the two groups
# Plot the distribution of ROE accross different status (Default=0; Default=1)
ggplot(data,aes(EBITDA,fill=Flag))+geom_density(alpha=0.2)



############Financial Learning#########

library(tidyverse)
library(caret)
library(randomForest)
library(neuralnet)
library(pROC)

# Transform "Default" and "Loss" variables to factors
data$Flag<-as.factor(data$Flag)
data$Loss_dummy<-as.factor(data$Loss_dummy) ##questo???###

# Split the dataset into training and testing samples 
# Sampling: a) Random Sampling and b) Stratified Sampling 

# a) Random Sampling 
# Set a random seed so that your results can be reproduced

set.seed(123)

perc<-0.7
n_train<-round(perc*nrow(data))
data_sample<-data[sample(nrow(data)),]          
data.train<-data_sample[1:n_train,]              
data.test<-data_sample[(n_train+1):nrow(data_sample),]    


summary(trainData)
summary(testData)



# With a simple random sampling technique, we are not sure whether 
# the subgroups of the Default variable are represented equally or proportionately within the two sub-samples.

# b) Stratified Sampling 
set.seed(300)
                            
div<-createDataPartition(y=data$Flag,p=perc,list=F)
   
# Training Sample
data.train_1<-data[div,] # 70% here
percentage(data.train_1$Flag)

# Test Sample
data.test_1<-data[-div,] # the rest of the data goes here
percentage(data.test_1$DFlag) # now the percentage of defaults in the two sub-samples is quite similar 


###### 1. REGRESSIONE LOGISTICA

### MODEL WITH ALL VARIABLES (for the time being)
fit1<-glm(Flag~.,data=data.train_1,family=binomial())
summary(fit1)

## Odds ratios 
exp(coefficients(fit1)) # how do you interpret the exp of coefficients?

## Get predicted default probabilities
data.test_1$score<-predict(fit1,type='response',data.test_1)

# Decide a cut-off and get predictions
cut_off<-def_perc
data.test_1$pred<-ifelse(data.test_1$score<=cut_off,0,1)

# Does the model classify the companies well?
# Let's introduce some measures used to investigate the performance of models for binary variables:  
# false positive rate and false negative rate, sensitivity and specificity
# you will see them in more detail during next lesson

##false positive rate
n_neg<-nrow(data.test_1[data.test_1$Default=='0',])
data.test_1$fp_flag<-ifelse(data.test_1$pred==1 & data.test_1$Default=='0',1,0)
fpr<-sum(data.test_1$fp_flag)/n_neg #false positive rate

##false negative rate
n_pos<-nrow(data.test_1[data.test_1$Flag=='1',])
data.test_1$fn_flag<-ifelse(data.test_1$pred==0 & data.test_1$Flag=='1',1,0)
fnr<-sum(data.test_1$fn_flag)/n_pos #false negative rate

##sensitivity
1-fnr

##specificity
1-fpr




##### 2. RANDOM FOREST


trainData$Flag <- factor(trainData$Flag, levels = c(0, 1))
testData$Flag <- factor(testData$Flag, levels = c(0, 1))


set.seed(123)
rf_model <- randomForest(Flag ~ ., data = trainData, ntree = 100)


rf_pred_prob <- predict(rf_model, newdata = testData, type = "prob")[,2]

rf_class <- ifelse(rf_pred_prob > 0.5, 1, 0)


confusionMatrix(factor(rf_class), factor(testData$Flag))

roc_curve_rf <- roc(as.numeric(as.character(testData$Flag)), rf_pred_prob)
plot(roc_curve_rf, main = "ROC - Random Forest")



#### 3. NEURAL NETWORK

# Convert Flag in numeric variable 
trainData$Flag <- as.numeric(trainData$Flag)
testData$Flag <- as.numeric(testData$Flag)


feature_names <- paste(names(trainData)[!names(trainData) %in% "Flag"], collapse = " + ")
formula_nn <- as.formula(paste("Flag ~", feature_names))


nn_model <- neuralnet(formula_nn, data = trainData, hidden = c(5, 3), linear.output = FALSE)
plot(nn_model)


nn_pred <- compute(nn_model, testData[, setdiff(names(testData), "Flag")])$net.result
nn_class <- ifelse(nn_pred > 0.5, 1, 0)


confusionMatrix(factor(nn_class), factor(testData$Flag))
roc_curve_nn <- roc(testData$Flag, nn_pred)
plot(roc_curve_nn, main = "ROC - Neural Network")


##########Models Confront
#Confront a AUC for all models
auc_logistic <- auc(roc_curve_logistic)
auc_rf <- auc(roc_curve_rf)
auc_nn <- auc(roc_curve_nn)

print(paste("AUC - Logistic Regression:", auc_logistic))
print(paste("AUC - Random Forest:", auc_rf))
print(paste("AUC - Neural Network:", auc_nn))



#########################Bayesian Learning######################################################
