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
