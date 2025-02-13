# Statistical-Learning-Exam


## Table of Contents  
1. [Introduction](#introduction)  
2. [Dataset](#dataset)  
3. [EDA (Exploratory Data Analysis)](#eda-exploratory-data-analysis)  
4. [Financial Learning](#financial-learning)  
5. [Bayesian Learning](#bayesian-learning)  
6. [Conclusions](#conclusions)  

## Introduction  

Title, Author
2. Your thesis (research question)
3. Models employed (and related R packages) 4. Main empirical findings

## Dataset  
The dataset includes financial indicators (in standardised values) for a sample of 2049 Italian small-medium enterprises (SMEs) in 2018 and information about their status (Flag variable; 0=Active, 1=Defaulted) one year later.

The variables of the dataset:
- Flag  
- Total_assets  
- Shareholders_funds  
- Long_term_debt  
- Loans  
- Turnover  
- EBITDA  
- Net_income  
- Tangible_fixed_assets  
- Profit_Loss_after_tax  
- Current_liabilities  
- Current_assets  
- Net_income_on_Total_Assets  
- Leverage  
- Tangible_on_total_assets  



## EDA (Exploratory Data Analysis)  
Cheked with command "print(sum(is.na(data)))" if there are missing values.
The result is 0 missing values.

with the command: "summary(data)" we anaylze our variables:

| Statistic   | Flag     | Total_assets | Shareholders_funds | Long_term_debt | Loans      | Turnover   | EBITDA     | Net_income  | Tangible_fixed_assets | Profit_Loss_after_tax | Current_liabilities | Current_assets | Net_income_on_Total_Assets | Leverage   | Tangible_on_total_assets |
|-------------|----------|--------------|--------------------|----------------|------------|------------|------------|--------------|-----------------------|-----------------------|--------------------|----------------|---------------------------|------------|--------------------------|
| Min.        | 0.00000  | -0.379995     | -0.31611            | -0.273864       | -0.287190  | -0.54400   | -1.48038   | -4.44506     | -0.343716              | -4.748423              | -0.409138           | -0.54253        | -8.300345                  | -0.589090  | -0.932028                |
| 1st Qu.     | 0.00000  | -0.242710     | -0.27329            | -0.273864       | -0.287190  | -0.21123   | -0.32716   | -0.22537     | -0.325433              | -0.233488              | -0.251286           | -0.31731        | -0.321765                  | -0.589090  | -0.790738                |
| Median      | 0.00000  | -0.151646     | -0.20815            | -0.273864       | -0.281337  | -0.09581   | -0.18035   | -0.15583     | -0.260832              | -0.158837              | -0.141358           | -0.16175        | -0.170741                  | -0.589090  | -0.375594                |
| Mean        | 0.06833  | -0.014183     | -0.01096            | -0.005915       | -0.019908  | -0.02592   | -0.01793   | -0.01425     | -0.000627              | -0.007344              | -0.008854           | -0.01445        |  0.009819                  |  0.006074  |  0.006065                |
| 3rd Qu.     | 0.00000  | -0.004814     | -0.02641            | -0.076514       |  0.006891  |  0.07194   |  0.08090   |  0.05976     | -0.045495              |  0.072584              |  0.012002           |  0.04629        |  0.256082                  |  0.235578  |  0.543831                |
| Max.        | 1.00000  | 18.908105     | 17.51151            | 15.399444       | 24.976590  | 26.97940   | 21.74522   | 15.79219     | 15.414282              | 16.601217              | 20.834466           | 24.83200        |  3.409903                  |  4.072354  |  3.114502                |

After that visualize the distribution of each variable



## Financial Learning  
...  

## Bayesian Learning  
...  

## Conclusions  
...  
