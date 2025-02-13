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

## Financial Learning  
...  

## Bayesian Learning  
...  

## Conclusions  
...  
