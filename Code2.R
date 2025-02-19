rm(list = ls())

   
library(readxl)  # For reading Excel files
library(ggplot2) # For data visualization
library(dplyr)   # For data manipulation
library(tidyr)   # For reshaping data



library(caret)
library(data.table)


library(rpart) 
library(partykit) 
library(rattle) 
library(rpart.plot) 
library(ROCR) 
library(randomForest) 


dataset_path <- "/Users/pedrocchiedoardo/Desktop/esame statistical learning/Dataset2_Companies.xlsx"


data <- read_excel(dataset_path)
data<-as.data.frame(data)



library(nnet)
library(neuralnet)


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


for (col in names(data)) {
  print(ggplot(data, aes(x = .data[[col]])) +
    geom_histogram(bins = 30, fill = "blue", alpha = 0.6) +
    labs(title = col, x = col, y = "Frequenza") +
    theme_minimal())
  Sys.sleep(1)  # Pausa di 1 secondo tra i grafici


for (col in names(data)) {
  print(ggplot(data, aes(y = .data[[col]])) +
    geom_boxplot(fill = "blue", alpha = 0.6) +
    labs(title = col, y = col) +
    theme_minimal())
  Sys.sleep(1)  # Pausa di 1 secondo tra i grafici


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




###########model selection

## Stepwise regression
fit_step<-step(fit1,direction='both')
summary(fit_step)

##ANOVA
anova(fit_step,fit1,test="Chisq")

## Get predicted default probabilities
data.test$score<-predict(fit_step,type='response',data.test)
data.test$score1<-predict(fit1,type='response',data.test)

## Plot AUROC
perf_auroc<-performance(prediction(data.test$score,data.test$Default),"auc")
auroc<-as.numeric(perf_auroc@y.values)

perf_plot<-performance(prediction(data.test$score,data.test$Default),"tpr","fpr")
plot(perf_plot,main='ROC',col='blue',lwd=2)

## Compare AUROC
### note: in this case the two AUROCs are very close, so the two ROC curves are overlapping
perf_auroc1<-performance(prediction(data.test$score1,data.test$Default),"auc")
auroc1<-as.numeric(perf_auroc1@y.values)

perf_plot1<-performance(prediction(data.test$score1,data.test$Default),"tpr","fpr")

plot(perf_plot,main='ROC',col='blue',lwd=2)
plot(perf_plot1,add=TRUE,col='red',lwd=2) 
legend("right",legend=c("Model from stepwise","Full model"),lty=(1:1),col=c("blue","red"))




######tree 
# We use the CART decision tree algorithm
# The CART algorithm for classification trees minimizes the Gini impurity in each group
fit1<-rpart(Flag~.,data=data.train,method="class")

# Print tree detail
printcp(fit1)

# Plot the tree
plot(fit1,margin=0.2,main="Tree: Recursive Partitioning")
text(fit1,cex=0.8) 

prp(fit1,type=2,extra=1,main="Tree: Recursive Partitioning") # type=2 draws the split labels below the node labels
                                                             # extra=1 displays the number of observations that fall in the node 
fancyRpartPlot(fit1)

# Make predictions on the test sample
data.test$fit1_score<-predict(fit1,type='prob',data.test)
fit1_pred<-prediction(data.test$fit1_score[,2],data.test$y)
fit1_perf<-performance(fit1_pred,"tpr","fpr")

# Model performance plot
plot(fit1_perf,lwd=2,colorize=TRUE,main="ROC Fit1: Recursive Partitioning")
lines(x=c(0,1),y=c(0,1),col="red",lwd=1,lty=3)







# AUROC, KS and GINI
# The KS statistic is the maximum difference between the cumulative percentage of "yes" (cumulative true positive rate)
# and the cumulative percentage of "no" (cumulative false positive rate)
# The Gini coefficient is measured in values between 0 and 1, where a score of 1 means that the model is 100% accurate
# in predicting the outcome, While a Gini score equal to 0 means that the model is entirely inaccurate (random model).

###cambiare i valori
fit1_AUROC<-round(performance(fit1_pred,measure="auc")@y.values[[1]]*100,2)
fit1_KS<-round(max(attr(fit1_perf,'y.values')[[1]]-attr(fit1_perf,'x.values')[[1]])*100,2)
fit1_Gini<-(2*fit1_AUROC-100)
CART_tree<-cat("AUROC:",fit1_AUROC,"KS:",fit1_KS,"Gini:",fit1_Gini)

# Conditional inference tree --> 
# Both rpart and ctree recursively perform univariate splits 
# of the target variable based on values on the other variables in the dataset
# Differently from the CART, ctree uses a significance test procedure 
# in order to select variables instead of selecting the variables that minimize the Gini impurity.
fit2<-ctree(Flag~.,data=data.train)
fit2
summary(fit2)


# This is essentially a decision tree but with extra information in the terminal nodes.
plot(fit2,gp=gpar(fontsize=6),ip_args=list(abbreviate=FALSE,id=FALSE))




# Make predictions on the test sample
data.test$fit2_score<-predict(fit2,type='prob',data.test)
fit2_pred<-prediction(data.test$fit2_score[,2],data.test$y)
fit2_perf<-performance(fit2_pred,"tpr","fpr")

# Model performance plot
plot(fit2_perf,lwd=2,colorize=TRUE,main="ROC Fit2: Conditional Inference Tree")
lines(x=c(0,1),y=c(0,1),col="red",lwd=1,lty=3)

#  AUROC, KS and GINI
fit2_AUROC<-round(performance(fit2_pred, measure = "auc")@y.values[[1]]*100,2)
fit2_KS<-round(max(attr(fit2_perf,'y.values')[[1]]-attr(fit2_perf,'x.values')[[1]])*100,2)
fit2_Gini<-(2*fit2_AUROC-100)
cond_inf_tree<-cat("AUROC:",fit2_AUROC,"KS:",fit2_KS,"Gini:",fit2_Gini)




##### 2. RANDOM FOREST

fit3<-randomForest(Flag~.,data=data.train,na.action=na.roughfix)

fit3_fitForest<-predict(fit3,newdata=data.test,type="prob")[,2]
fit3_fitForest.na<-as.data.frame(cbind(data.test$y,fit3_fitForest))
colnames(fit3_fitForest.na)<-c('y','pred')
fit3_fitForest.narm<-as.data.frame(na.omit(fit3_fitForest.na)) # remove NA (missing values)

fit3_pred<-prediction(fit3_fitForest.narm$pred,fit3_fitForest.narm$y)
fit3_perf<-performance(fit3_pred,"tpr","fpr")

#Plot variable importance
varImpPlot(fit3,main="Random Forest: Variable Importance")

# Model Performance plot
plot(fit3_perf,colorize=TRUE,lwd=2,main="fit3 ROC: Random Forest",col="blue")
lines(x=c(0,1),y=c(0,1),col="red",lwd=1,lty=3)

# AUROC, KS and GINI
##cambiare valori
fit3_AUROC<-round(performance(fit3_pred, measure="auc")@y.values[[1]]*100,2)
fit3_KS<-round(max(attr(fit3_perf,'y.values')[[1]]-attr(fit3_perf,'x.values')[[1]])*100,2)
fit3_Gini<-(2*fit3_AUROC-100)
rand_for<-cat("AUROC:",fit3_AUROC,"KS:",fit3_KS,"Gini:",fit3_Gini)

#Compare ROC Performance of the 3 models
plot(fit1_perf,col='blue',lty=1,main='ROCs: Model Performance Comparison') # Recursive partitioning
plot(fit2_perf,col='gray',lty=2,add=TRUE); # Conditional inference tree
plot(fit3_perf, col='red',lty=3,add=TRUE); # Random forest
lines(c(0,1),c(0,1),col= "green",lty=4) # random line

legend(0.6,0.5, c('Fit1: Recursive partitioning (AUROC=79.85)','Fit2: Conditional inference tree (AUROC=91.86)','Fit3: Random forest (AUROC=79.3)', 'Random line'),
col=c('blue','grey','red','green'),lwd=c(1,2,3), cex=0.7)




#### 3. NEURAL NETWORK
###########################
##### NEURAL NETWORK #####
###########################

# Clear the environment to remove all existing objects
rm(list=ls())

# Install necessary packages (remove the # to install them if needed)
#install.packages("readr")  # For reading CSV files
install.packages("nnet")    # For neural networks
install.packages("neuralnet")  # For training deep learning models

# Load required libraries
library(readr)     # Load readr for data handling
library(nnet)      # Load nnet for neural network functions
library(neuralnet) # Load neuralnet for training neural networks

#####################################
########## APPLICATION FLAG #########
#####################################

# Import dataset
# Assuming "data.csv" contains the dataset with the "Flag" column
# data <- read_csv("path/data.csv")

# Convert dataset to a dataframe for easy manipulation
data <- as.data.frame(data)

# Generate summary statistics of the dataset
summary(data)

# Show frequency of each class in the Flag column
table(data$Flag)

# Encode the binary dependent variable (Flag) into dummy variables
train <- cbind(data[, -which(names(data) == "Flag")], class.ind(as.factor(data$Flag)))

# Rename columns: Keep the original feature names and add "F0" and "F1" for the encoded target variable
names(train) <- c(names(data)[-which(names(data) == "Flag")], "F0", "F1")

# Train a neural network model on the dataset
set.seed(123)  # Set seed for reproducibility
nn <- neuralnet(F0 + F1 ~ ., data=train,
                hidden=c(5),  # Define one hidden layer with 5 neurons
                act.fct="logistic",  # Use logistic activation function
                err.fct='ce',  # Use cross-entropy error function
                linear.output=FALSE,  # Output is categorical, not continuous
                lifesign="minimal")  # Show minimal training progress output

# Plot the trained neural network model
plot(nn)

##### Compute predictions #####
# Generate predictions on the training dataset
nn_pred <- compute(nn, train[, 1:(ncol(train) - 2)])

# Extract predicted results
nn_pred1 <- nn_pred$net.result
head(nn_pred1)  # Display first few predictions

# Calculate in-sample accuracy
original_values <- max.col(train[, (ncol(train) - 1):ncol(train)])  # Extract actual class labels
predicted <- max.col(nn_pred1)  # Get predicted class labels
accuracy <- mean(predicted == original_values)  # Compute accuracy
print(accuracy)  # Print the accuracy of the model

##### Cross-validation #####
# Initialize an empty vector to store accuracy results
outs <- NULL

# Set proportion for train-test split
proportion <- 0.80  # 80% training, 20% testing

# Define number of folds for cross-validation
k <- 10  # 10-fold cross-validation
set.seed(123)  # Set seed for reproducibility

# Perform k-fold cross-validation
for(i in 1:k) {
  index <- sample(1:nrow(train), round(proportion * nrow(train)))  # Randomly select training indices
  train_cv <- train[index, ]  # Create training subset
  test_cv <- train[-index, ]  # Create testing subset
  
  # Train neural network on training subset
  nn_cv <- neuralnet(F0 + F1 ~ ., data=train_cv,
                     hidden=c(5),  # Use the same hidden layer configuration
                     act.fct="logistic",  # Use logistic activation function
                     err.fct="ce",  # Use cross-entropy error function
                     linear.output=FALSE)  # Categorical output
  
  # Generate predictions on the test set
  nn_pred <- compute(nn_cv, test_cv[, 1:(ncol(train) - 2)])
  
  # Extract predicted results
  nn_pred1 <- nn_pred$net.result
  
  # Evaluate accuracy
  original_values <- max.col(test_cv[, (ncol(train) - 1):ncol(train)])  # Extract actual class labels
  predicted <- max.col(nn_pred1)  # Get predicted class labels
  outs[i] <- mean(predicted == original_values)  # Store accuracy result
}

# Compute and print the mean accuracy across all folds
print(mean(outs))





#####algortmo moificato, quello sopra non è di classficiazione
# Train a neural network model on the dataset
nn <- neuralnet(Flag ~ ., data=train,
                hidden=c(5),  # Define one hidden layer with 5 neurons
                act.fct="logistic",  # Use logistic activation function
                err.fct='ce',  # Use cross-entropy error function
                linear.output=FALSE,  # Output is categorical, not continuous
                lifesign="minimal")  # Show minimal training progress output

# Plot the trained neural network model
plot(nn)

##### Compute predictions #####
# Generate predictions on the test dataset
nn_pred <- compute(nn, test[, -which(names(test) == "Flag")])

# Extract predicted results
nn_pred1 <- nn_pred$net.result

# Convert predictions to class labels
predicted <- ifelse(nn_pred1 > 0.5, 1, 0)

# Evaluate accuracy
accuracy <- mean(predicted == as.numeric(as.character(test$Flag)))  # Compute accuracy
print(accuracy)  # Print the accuracy of the model

##### Cross-validation #####
# Initialize an empty vector to store accuracy results
outs <- NULL

# Define number of folds for cross-validation
k <- 10  # 10-fold cross-validation
set.seed(123)  # Set seed for reproducibility

# Perform k-fold cross-validation
for(i in 1:k) {
  index <- sample(1:nrow(train), round(0.8 * nrow(train)))  # Randomly select training indices
  train_cv <- train[index, ]  # Create training subset
  test_cv <- train[-index, ]  # Create testing subset
  
  # Train neural network on training subset
  nn_cv <- neuralnet(Flag ~ ., data=train_cv,
                     hidden=c(5),  # Use the same hidden layer configuration
                     act.fct="logistic",  # Use logistic activation function
                     err.fct="ce",  # Use cross-entropy error function
                     linear.output=FALSE)  # Categorical output
  
  # Generate predictions on the test set
  nn_pred <- compute(nn_cv, test_cv[, -which(names(test_cv) == "Flag")])
  
  # Extract predicted results
  nn_pred1 <- nn_pred$net.result
  
  # Convert predictions to class labels
  predicted <- ifelse(nn_pred1 > 0.5, 1, 0)
  
  # Evaluate accuracy
  outs[i] <- mean(predicted == as.numeric(as.character(test_cv$Flag)))  # Store accuracy result
}

# Compute and print the mean accuracy across all folds
print(mean(outs))


manca safe ai


#########################Bayesian Learning######################################################

LOgistic regression

Bayesian networks

