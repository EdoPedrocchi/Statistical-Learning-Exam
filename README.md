[LINK](https://edopedrocchi.github.io/Statistical-Learning-Exam/)
# Statistical-Learning-Exam


## Table of Contents  
1. [Introduction](#introduction)  
2. [Dataset](#dataset)  
3. [EDA (Exploratory Data Analysis)](#eda-exploratory-data-analysis)  
4. [Financial Learning](#financial-learning)  
5. [Bayesian Learning](#bayesian-learning)  
6. [Conclusions](#conclusions)  

## Introduction  

The project aims to predict the default of Italian Small and Medium Enterprises (SMEs) using various statistical learning techniques. The primary research question guiding this analysis is:  

**"Which statistical learning approach provides the most accurate and reliable predictions for SME default risk in Italy?"**  

To address this, the project evaluates and compares 2 approaches:
1. Financial Learning (Frequentist approach)
2. Bayesian Learning

Describe In general financial Learning and Bayesian learning

Find research topic similar

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

  PUORUI CREARE NUOVI INDICATORI?

DESCRIBE EACH VARIABLE WHAT REPRESENT


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


![f95f9db8-6b33-4c61-a7b5-9a0b30938a08](https://github.com/user-attachments/assets/9764f08b-5bd9-48f2-b95a-13d544f2aa8d)
![f2fcd318-097c-4e1c-9197-98f5db3a6913](https://github.com/user-attachments/assets/6438e391-8e53-4325-84b9-9c512d871778)
![d34a7875-8681-4022-8c83-12171c0838c1](https://github.com/user-attachments/assets/825215e5-0830-41b6-aaef-e75d962eff44)
![c25421c0-2d63-431a-b042-856988d69e3a](https://github.com/user-attachments/assets/bc23825c-527b-4cd6-b50c-b865f587c2ec)
![c15dc0d6-2da8-408e-b9ff-5e54a093f0b9](https://github.com/user-attachments/assets/e14757c0-96e2-4d92-90aa-d5d11e5ec5a9)
![b59b5fa2-2d2b-4020-bbe8-0abaab06d1ea](https://github.com/user-attachments/assets/7d62ac40-9ec2-4205-8c4a-04c40bfd4e97)
![2514110d-2ae9-4339-bdf5-02d212177fd1](https://github.com/user-attachments/assets/16539c43-9e10-4000-9aac-26ade64f10fc)
![902f4ab1-7490-493b-a225-b421a2fbd38d](https://github.com/user-attachments/assets/167db195-66ad-4ef9-8534-2973a21dbf19)
![210f7ea3-ecf4-4922-8ad7-aae32d410004](https://github.com/user-attachments/assets/3189137e-e74b-47eb-962b-f259097e0aed)
![160f146e-4a13-4e8c-9dc8-3c1cf1d10195](https://github.com/user-attachments/assets/83c7c4ca-3ba0-4c3f-9f1f-459152ee6077)
![19a5b473-1640-4779-9a39-ae4ccdd0699a](https://github.com/user-attachments/assets/e42cfa9a-4ace-4d5d-8ecd-564a091106ba)
![8a08fcfe-4ddb-406b-837c-b76572c95497](https://github.com/user-attachments/assets/78722776-38a3-4493-ab73-740db5a3b8cf)
![3de430e4-0729-4c05-ba5b-6f3ef6ab3d9b](https://github.com/user-attachments/assets/9e2955b0-d93e-44a1-9cd9-a35f15ac4e44)
![0cc02b3e-60fa-4812-a873-a4bae6e0103c](https://github.com/user-attachments/assets/17a939a1-a9a0-4556-afdc-1e536d35acb7)
![0a9c50fd-38f3-4a7c-9be2-6c85b5c7d3cb](https://github.com/user-attachments/assets/3dea6d6d-d6a1-409f-8e80-d4207003af65)


Now boxplots

![fff2aec3-1397-42e2-8525-80f8f6633ca7](https://github.com/user-attachments/assets/288fd60a-2e79-4b7f-a2c0-5a9cee8fe662)
![eec71af9-66e7-425f-9efc-c903757e3283](https://github.com/user-attachments/assets/21d1ec4b-bdd4-4d70-a8a3-b906aa2845de)
![eaacdc39-4ca8-4068-82c5-d83cde696ec4](https://github.com/user-attachments/assets/e67109e1-6939-48be-b44e-e6ef45966aa5)
![e830774e-74e6-410c-8216-78a6bff54e2b](https://github.com/user-attachments/assets/fb47d053-eaba-4418-a470-6f1720f80aeb)
![dabd14a2-b5e5-426a-8bc2-4bcbba9305d9](https://github.com/user-attachments/assets/c97d3b38-796c-4c9a-b06c-42172a434645)
![c8ae6f4b-9bfa-453a-b3a0-2b1d7938d4fa](https://github.com/user-attachments/assets/7561e837-6ad9-4be4-9c9d-5f46ee21b709)
![a81fc231-7737-4d08-b514-421a4ac0a8c3](https://github.com/user-attachments/assets/a91fcc88-1bb5-48c2-a348-14f8874020c1)
![8928b899-81cd-4135-9c10-c388c1bb0912](https://github.com/user-attachments/assets/943bf576-a2f0-4506-ac57-fe288c824fe5)
![5598f254-4d55-40a7-bf22-ff73d66201b2](https://github.com/user-attachments/assets/11201d73-0002-4b88-9999-9bf6349fbbbe)
![919e8a45-6d1f-4fa4-95f2-eb4ece9a7233](https://github.com/user-attachments/assets/be03b9e2-255b-4b04-a9bb-35b0663467ae)
![98d8daac-f2c2-4e70-b8fa-d4e41ae5ca58](https://github.com/user-attachments/assets/5065c589-fe37-4331-8443-0c0b756264b3)
![40a7deef-d602-4059-bf10-3184681967e6](https://github.com/user-attachments/assets/39f4f485-1f7f-4149-bfd4-742d2a69a201)
![7da5027c-6f5f-4fbb-a889-584cc6e8c954](https://github.com/user-attachments/assets/41cfe1de-c716-4832-8806-546f7fb03a8b)
![2bf34b73-c2a4-4583-8128-fa8e7c62d4cd](https://github.com/user-attachments/assets/c269da41-ebe0-4254-be84-3eb1049a2730)
![1b7edd59-0eeb-4d2a-a828-0c02824e517c](https://github.com/user-attachments/assets/04cb49d5-596a-467f-9a74-54fae1d5a137)

Watch the cake of our target variable
| Status | Count |
|--------|-------|
| 0      | 1909  |
| 1      | 140   |

![2d243929-f63b-4d16-8f84-fb7e3d8a01ae](https://github.com/user-attachments/assets/32e60974-2dda-44a2-a52c-7a9d55852ac0)



Now watch the Correlation matrix
![407a8455-8dcc-4c39-83b5-e6acdec54770](https://github.com/user-attachments/assets/22ea6c9b-21e5-42a6-af13-38f3b8386147)


| **Variable**               | **Flag** | **Total_assets** | **Shareholders_funds** | **Long_term_debt** | **Loans** | **Turnover** | **EBITDA** | **Net_income** | **Tangible_fixed_assets** | **Profit_Loss_after_tax** | **Current_liabilities** | **Current_assets** | **Net_income_on_Total_Assets** | **Leverage** | **Tangible_on_total_assets** |
|----------------------------|----------|------------------|------------------------|--------------------|-----------|--------------|------------|----------------|----------------------------|--------------------------|------------------------|--------------------|--------------------------------|--------------|----------------------------|
| **Flag**                   | 1.000000 | 0.120135         | 0.065515               | 0.021874           | 0.066947  | 0.086075     | 0.066736   | -0.014107      | 0.050775                  | -0.014763                | 0.139092               | 0.087262            | -0.236280                    | -0.085759    | -0.062352                  |
| **Total_assets**           | 0.120135 | 1.000000         | 0.799717               | 0.487353           | 0.553106  | 0.543062     | 0.667485   | 0.271552       | 0.712671                  | 0.267494                | 0.915710               | 0.775949            | 0.001880                     | 0.086082     | 0.196559                  |
| **Shareholders_funds**     | 0.065515 | 0.799717         | 1.000000               | 0.327333           | 0.371205  | 0.410217     | 0.521549   | 0.513540       | 0.572102                  | 0.508739                | 0.555177               | 0.650968            | 0.091663                     | 0.009553     | 0.210004                  |
| **Long_term_debt**         | 0.021874 | 0.487353         | 0.327333               | 1.000000           | 0.330673  | 0.060256     | 0.170314   | -0.097026      | 0.531025                  | -0.097865               | 0.346402               | 0.396220            | -0.074976                    | 0.557805     | 0.271232                  |
| **Loans**                  | 0.066947 | 0.553106         | 0.371205               | 0.330673           | 1.000000  | 0.630621     | 0.405781   | 0.184351       | 0.228420                  | 0.180468                | 0.587200               | 0.680304            | -0.034765                    | 0.180990     | 0.073458                  |
| **Turnover**               | 0.086075 | 0.543062         | 0.410217               | 0.060256           | 0.630621  | 1.000000     | 0.645126   | 0.421704       | 0.120212                  | 0.419033                | 0.566604               | 0.649621            | 0.097107                     | -0.000610    | -0.012307                  |
| **EBITDA**                 | 0.066736 | 0.667485         | 0.521549               | 0.170314           | 0.405781  | 0.645126     | 1.000000   | 0.571742       | 0.369585                  | 0.573066                | 0.608392               | 0.513782            | 0.337960                     | 0.050950     | 0.157352                  |
| **Net_income**             | -0.014107| 0.271552         | 0.513540               | -0.097026          | 0.184351  | 0.421704     | 0.571742   | 1.000000       | -0.066144                 | 0.999946                | 0.126995               | 0.395780            | 0.544851                     | -0.048617    | -0.054412                  |
| **Tangible_fixed_assets**  | 0.050775 | 0.712671         | 0.572102               | 0.531025           | 0.228420  | 0.120212     | 0.369585   | -0.066144      | 1.000000                  | -0.067513               | 0.609736               | 0.297549            | -0.065170                    | 0.188769     | 0.553759                  |
| **Profit_Loss_after_tax**  | -0.014763| 0.267494         | 0.508739               | -0.097865          | 0.180468  | 0.419033     | 0.573066   | 0.999946       | -0.067513                 | 1.000000                | 0.123849               | 0.391167            | 0.548562                     | -0.048762    | -0.054582                  |
| **Current_liabilities**    | 0.139092 | 0.915710         | 0.555177               | 0.346402           | 0.587200  | 0.566604     | 0.608392   | 0.126995       | 0.609736                  | 0.123849                | 1.000000               | 0.719490            | -0.035749                    | 0.019858     | 0.095107                  |
| **Current_assets**         | 0.087262 | 0.775949         | 0.650968               | 0.396220           | 0.680304  | 0.649621     | 0.513782   | 0.395780       | 0.297549                  | 0.391167                | 0.719490               | 1.000000            | 0.044429                     | 0.046599     | -0.058964                 |
| **Net_income_on_Total_Assets** | -0.236280 | 0.001880 | 0.091663 | -0.074976 | -0.034765 | 0.097107 | 0.337960 | 0.544851 | -0.065170 | 0.548562 | -0.035749 | 0.044429 | 1.000000 | -0.060431 | -0.071141 |
| **Leverage**               | -0.085759 | 0.086082         | 0.009553               | 0.557805           | 0.180990  | -0.000610    | 0.050950   | -0.048617      | 0.188769                  | -0.048762               | 0.019858               | 0.046599            | -0.060431                    | 1.000000     | 0.374895                  |
| **Tangible_on_total_assets**| -0.062352 | 0.196559         | 0.210004               | 0.271232           | 0.073458  | -0.012307    | 0.157352   | -0.054412      | 0.553759                  | -0.054582               | 0.095107               | -0.058964           | -0.071141                    | 0.374895     | 1.000000                  |





## Financial Learning  

METTI LA TEORIA DI OGNI MODELLO 

METTERE SUMMARY DEI TEST E TRAINING SET

MEttere pro e contro di ogni modello
e la toeria dietro
   
1. **Logistic Regression**  

## Logistic Regression Model Evaluation

### Confusion Matrix

| Prediction | 0   | 1   |
|------------|-----|-----|
| **0**      | 578 | 30  |
| **1**      | 4   | 2   |

###  Metrics

| Metric                             | Value         |
|------------------------------------|---------------|
| **Accuracy**                       | 94.46%        |
| **95% Confidence Interval**        | (92.35%, 96.14%) |
| **No Information Rate**            | 94.79%        |
| **P-Value [Acc > NIR]**            | 0.6827        |
| **Kappa**                          | 0.0903        |
| **McNemar's Test P-Value**         | 1.807e-05     |
| **Sensitivity (Recall)**           | 99.31%        |
| **Specificity**                    | 6.25%         |
| **Positive Predictive Value (Precision)** | 95.07%   |
| **Negative Predictive Value**      | 33.33%        |
| **Prevalence**                     | 94.79%        |
| **Detection Rate**                 | 94.14%        |
| **Detection Prevalence**           | 99.02%        |
| **Balanced Accuracy**              | 52.78%        |


  
![fa3a99a1-9f84-4f07-9982-7216228b6872](https://github.com/user-attachments/assets/beb45a47-0b9d-4e02-82c3-7a30312951e2)





2. **Random Forest**  



### metrics 

| Metric                   | Value         |
|---------------------------|---------------|
| **Accuracy**              | 0.9642        |
| **95% CI**                | (0.9463, 0.9774) |
| **No Information Rate**   | 0.9479        |
| **P-Value [Acc > NIR]**   | 0.03683       |
| **Kappa**                 | 0.5745        |
| **Mcnemar's Test P-Value**| 0.05501       |
| **Sensitivity**           | 0.9897        |
| **Specificity**           | 0.5000        |
| **Pos Pred Value**        | 0.9730        |
| **Neg Pred Value**        | 0.7273        |
| **Prevalence**            | 0.9479        |
| **Detection Rate**        | 0.9381        |
| **Detection Prevalence**  | 0.9642        |
| **Balanced Accuracy**     | 0.7448        |

---

### Confusion Matrix

|               | **Reference: 0** | **Reference: 1** |
|---------------|------------------|------------------|
| **Prediction: 0** | 576              | 16               |
| **Prediction: 1** | 6                | 16               |

![7275d9c1-e89c-4fc5-b005-cf6f4e3763ba](https://github.com/user-attachments/assets/54de26c0-694f-4621-9c57-78442a3ca7f5)


5. **Neural Networks**  
 
[Quartz 4.pdf](https://github.com/user-attachments/files/18789959/Quartz.4.pdf)


### Confusion Matrix and Statistics

| Prediction | 0   | 1   |
|------------|-----|-----|
| **0**      | 0   | 0   |
| **1**      | 582 | 32  |

#### Metrics:

| Metric                        | Value        |
|-------------------------------|--------------|
| **Accuracy**                   | 0.0521       |
| **95% CI**                     | (0.0359, 0.0728) |
| **No Information Rate**        | 0.9479       |
| **P-Value [Acc > NIR]**        | 1            |
| **Kappa**                      | 0            |
| **Mcnemar's Test P-Value**     | <2e-16       |
| **Sensitivity**                | 0.00000      |
| **Specificity**                | 1.00000      |
| **Pos Pred Value**             | NaN          |
| **Neg Pred Value**             | 0.05212      |
| **Prevalence**                 | 0.94788      |
| **Detection Rate**             | 0.00000      |
| **Detection Prevalence**       | 0.00000      |
| **Balanced Accuracy**          | 0.50000      |
| **'Positive' Class**           | 0            |

![e3dcf7c8-8195-4f40-a7c1-ce7b0aa3285e](https://github.com/user-attachments/assets/9b16bd9b-1382-4218-b128-57c566c839d2)




## MOdels confront




## Bayesian Learning  

...  

## Conclusions  
...  
