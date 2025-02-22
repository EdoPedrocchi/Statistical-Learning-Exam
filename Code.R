rm(list = ls())

   
library(readxl)  # For reading Excel files
library(ggplot2) # For data visualization
library(dplyr)   # For data manipulation
library(tidyr)   # For reshaping data


dataset_path <- "/Users/pedrocchiedoardo/Desktop/esame statistical learning/Dataset2_Companies.xlsx"


data <- read_excel(dataset_path)

########################## EDA #####################################################


print(head(data))

data$ID <- NULL

print(colnames(data))
print(head(data))
print(sum(is.na(data))) 

# Convert data to long format for easier plotting
long_data <- data %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")


# Create a list of unique variables
unique_variables <- unique(long_data$Variable)

# Plot distributions of each variables
for (variable in unique_variables) {
  plot_data <- long_data %>% filter(Variable == variable)
  p <- ggplot(plot_data, aes(x = Value)) +
    geom_histogram(bins = 30, fill = "skyblue", color = "black", alpha = 0.7) +
    theme_minimal() +
    labs(title = paste("Distribution of", variable), 
         x = "Value", 
         y = "Frequency")
  print(p)
  readline(prompt = "Press [Enter] to see the next plot...")
}
str(data)

summary(data)

cor(data[, sapply(data, is.numeric)])

correlation_matrix <- cor(data[, sapply(data, is.numeric)])
print(correlation_matrix)

# Heatmap  correlation matrix
heatmap(correlation_matrix, main="Matrice di correlazione", col=heat.colors(10))


for (variable in unique_variables) {
  plot_data <- long_data %>% filter(Variable == variable)
  p <- ggplot(plot_data, aes(y = Value)) +
    geom_boxplot(fill = "orange", color = "black", alpha = 0.7) +
    theme_minimal() +
    labs(title = paste("Boxplot of", variable), 
         y = "Value")
  print(p)
  readline(prompt = "Press [Enter] to see the next plot...")
}

for (variable in unique_variables) {
  plot_data <- long_data %>% filter(Variable == variable)
  p <- ggplot(plot_data, aes(x = Value)) +
    geom_density(fill = "purple", alpha = 0.5) +
    theme_minimal() +
    labs(title = paste("Density Plot of", variable), 
         x = "Value", 
         y = "Density")
  print(p)
  readline(prompt = "Press [Enter] to see the next plot...")
}


library(ggplot2)

# highlits the flag variable, to watch in a cake the 0 and 1
flag_counts <- table(data$Flag)
flag_df <- as.data.frame(flag_counts)
colnames(flag_df) <- c("Status", "Count")

ggplot(flag_df, aes(x = "", y = Count, fill = factor(Status))) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar(theta = "y") +
  labs(title = "Distribuzione della variabile Flag", fill = "Status") +
  theme_minimal()

print(flag_df)

##### se vuoi verificare megli orelazione tra due variabili usa scatter plot


############Financial Learning#########

library(tidyverse)
library(caret)
library(randomForest)
library(neuralnet)
library(pROC)



set.seed(123)

# training (70%) e test (30%)
trainIndex <- createDataPartition(data$Flag, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

#standardize the features 
trainFlag <- trainData$Flag
testFlag <- testData$Flag


trainFeatures <- trainData[, setdiff(names(trainData), "Flag")]
testFeatures <- testData[, setdiff(names(testData), "Flag")]

preProcessValues <- preProcess(trainFeatures, method = c("center", "scale"))
trainFeatures <- predict(preProcessValues, trainFeatures)
testFeatures <- predict(preProcessValues, testFeatures)

trainData <- cbind(trainFeatures, Flag = trainFlag)
testData <- cbind(testFeatures, Flag = testFlag)


summary(trainData)
summary(testData)


###### 1. REGRESSIONE LOGISTICA

logistic_model <- glm(Flag ~ ., data = trainData, family = "binomial")
logistic_pred <- predict(logistic_model, newdata = testData, type = "response")
logistic_class <- ifelse(logistic_pred > 0.5, 1, 0)

confusionMatrix(factor(logistic_class), factor(testData$Flag))
roc_curve_logistic <- roc(testData$Flag, logistic_pred)
plot(roc_curve_logistic, main = "ROC - Logistic Regression")


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


