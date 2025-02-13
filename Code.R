rm(list = ls())

   
# Load required libraries
library(readxl)  # For reading Excel files
library(ggplot2) # For data visualization
library(dplyr)   # For data manipulation
library(tidyr)   # For reshaping data

# Set the path to your dataset file
dataset_path <- "/Users/pedrocchiedoardo/Desktop/esame statistical learning/Dataset2_Companies.xlsx"

# Import the dataset
data <- read_excel(dataset_path)

########################## EDA #####################################################

# Display the first few rows of the dataset
print(head(data))

data$ID <- NULL

print(colnames(data)) # Get the names of all columns in the dataset
print(head(data))
print(sum(is.na(data))) # Check  missing values

# Convert data to long format for easier plotting
long_data <- data %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")

#

# Create a list of unique variables
unique_variables <- unique(long_data$Variable)

# Plot each variable one at a time
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

# Heatmap della matrice di correlazione
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

# Creare un dataframe con i conteggi
flag_counts <- table(data$Flag)
flag_df <- as.data.frame(flag_counts)
colnames(flag_df) <- c("Status", "Count")

# Creare il grafico a torta
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

# Dividi il dataset in training (70%) e test (30%)
trainIndex <- createDataPartition(data$Flag, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Salva la variabile target
trainFlag <- trainData$Flag
testFlag <- testData$Flag

# Rimuovi la variabile target per standardizzare solo le feature
trainFeatures <- trainData[, setdiff(names(trainData), "Flag")]
testFeatures <- testData[, setdiff(names(testData), "Flag")]

# Standardizza le feature
preProcessValues <- preProcess(trainFeatures, method = c("center", "scale"))
trainFeatures <- predict(preProcessValues, trainFeatures)
testFeatures <- predict(preProcessValues, testFeatures)

# Riaggiungi la variabile target intatta
trainData <- cbind(trainFeatures, Flag = trainFlag)
testData <- cbind(testFeatures, Flag = testFlag)


summary(trainData)
summary(testData)

# ---------------------------
# 1. REGRESSIONE LOGISTICA
# ---------------------------
logistic_model <- glm(Flag ~ ., data = trainData, family = "binomial")
logistic_pred <- predict(logistic_model, newdata = testData, type = "response")
logistic_class <- ifelse(logistic_pred > 0.5, 1, 0)

# Valutazione del modello
confusionMatrix(factor(logistic_class), factor(testData$Flag))
roc_curve_logistic <- roc(testData$Flag, logistic_pred)
plot(roc_curve_logistic, main = "ROC - Logistic Regression")

# ---------------------------
# 2. RANDOM FOREST
# ---------------------------
trainData$Flag <- factor(trainData$Flag, levels = c(0, 1))
rf_model <- randomForest(Flag ~ ., data = trainData, ntree = 100)
rf_pred <- predict(rf_model, newdata = testData, type = "response")
rf_class <- ifelse(rf_pred == 1, 1, 0)
confusionMatrix(factor(rf_class), factor(testData$Flag))
# Valutazione del modello
confusionMatrix(factor(rf_class), factor(testData$Flag))
roc_curve_rf <- roc(testData$Flag, rf_pred)
plot(roc_curve_rf, main = "ROC - Random Forest")

# ---------------------------
# 3. NEURAL NETWORK
# ---------------------------
# Converti Flag in variabile numerica per neuralnet
trainData$Flag <- as.numeric(trainData$Flag)
testData$Flag <- as.numeric(testData$Flag)

# Crea una formula dinamica per neuralnet
feature_names <- paste(names(trainData)[!names(trainData) %in% "Flag"], collapse = " + ")
formula_nn <- as.formula(paste("Flag ~", feature_names))

# Addestra la Neural Network
nn_model <- neuralnet(formula_nn, data = trainData, hidden = c(5, 3), linear.output = FALSE)
plot(nn_model)

# Fai le previsioni con la Neural Network
nn_pred <- compute(nn_model, testData[, setdiff(names(testData), "Flag")])$net.result
nn_class <- ifelse(nn_pred > 0.5, 1, 0)

# Valutazione del modello
confusionMatrix(factor(nn_class), factor(testData$Flag))
roc_curve_nn <- roc(testData$Flag, nn_pred)
plot(roc_curve_nn, main = "ROC - Neural Network")

# ---------------------------
# CONFRONTO DEI MODELLI
# ---------------------------
# Calcola e confronta l'AUC per tutti i modelli
auc_logistic <- auc(roc_curve_logistic)
auc_rf <- auc(roc_curve_rf)
auc_nn <- auc(roc_curve_nn)

print(paste("AUC - Logistic Regression:", auc_logistic))
print(paste("AUC - Random Forest:", auc_rf))
print(paste("AUC - Neural Network:", auc_nn))



