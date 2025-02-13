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






######################### bayes ########
library(e1071)
library(recipes)
library(lattice)
library(caret)



data$Flag <- as.factor(data$Flag)  # Convertire la variabile target in fattore

# Dividere i dati in training e test
set.seed(123)
train_ratio <- 0.8
train_index <- sample(seq_len(nrow(data)), size = floor(train_ratio * nrow(data)))

# Create training and test sets
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Modello Naïve Bayes
model <- naiveBayes(Flag ~ ., data = train_data)

# Predizioni
y_pred <- predict(model, test_data)



# Create a confusion matrix manually using table()
conf_matrix <- table(Predicted = y_pred, Actual = test_data$Flag)

# Print the confusion matrix
print(conf_matrix)


# Visualizzazione delle probabilità
probabilities <- predict(model, test_data, type = "raw")
test_data$Predicted_Prob <- probabilities[, 2]

ggplot(test_data, aes(x = Predicted_Prob, fill = Flag)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 20) +
  labs(title = "Distribuzione delle Probabilità Predette",
       x = "Probabilità di Default",
       y = "Frequenza") +
  theme_minimal()



x<-rnorm(100)
par(mfrow=c(2,1))
theta=0.9
z<-rep(0,101)
for (i in 1:100) {z[i+1]<-(theta*z[i]+x[i])}; plot(z,type=’l’)
theta<- -theta
z<-rep(0,101)
for (i in 1:100) {z[i+1]<-(theta*z[i]+x[i])}; plot(z,type=’l’)





set.seed(154) # So that we can reproduce the results
w = rnorm(200); x = cumsum(w)
wd = w +.2; xd = cumsum(wd)
# We set a limit for y considering all the lines
plot(xd, ylim=c(-5,55), main="random walk", ylab="")
# Different colour for the curve we are adding
lines(x, col=4);
# Different symbol for the straight lines
abline(h=0, col=4, lty=2); abline(a=0, b=.2, lty=2)





