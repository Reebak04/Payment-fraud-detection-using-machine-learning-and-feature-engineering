# Payment-fraud-detection-using-machine-learning-and-feature-engineering


Payment Fraud Detection using Machine Learning and Feature Engineering
Table of Contents
Project Overview
Motivation
Dataset
### Introduction
In recent years, the rise of digital transactions has significantly increased the need for effective payment fraud detection systems. Fraudulent activities not only lead to financial losses for businesses but also undermine consumer trust in online payment systems. This project aims to address these challenges by leveraging machine learning techniques to develop a robust payment fraud detection model.

Using a dataset containing transaction records, we explore various machine learning algorithms, focusing on feature engineering and model evaluation to identify fraudulent transactions. The primary goal is to create a model that accurately classifies transactions as either fraudulent or legitimate, thereby enhancing the security and integrity of online payments.

This project employs several machine learning models, including Logistic Regression, Decision Trees, and Random Forest, to analyze patterns in the data. Additionally, visualization techniques are utilized to present model performance metrics, making it easier to interpret the results.

By developing a reliable fraud detection system, we aim to contribute to the ongoing efforts in safeguarding digital transactions and promoting secure online commerce.
### Approach

1. Data Preprocessing
2. Feature Engineering
3. Model Building
4. Model Evaluation
   

### Motivation
Payment fraud detection is a real-world problem that impacts millions of financial transactions daily. We aim to build a reliable machine learning pipeline to identify suspicious transactions, focusing on minimizing false negatives (failing to detect fraud) while maintaining accuracy.

### Dataset
The dataset contains anonymized features of transactions, such as time of transaction, amount, and other behavioral and transactional features. A typical dataset will have:

Transaction Amount: Amount involved in the transaction.
Transaction Time: Time of the transaction.
Customer Behavior Features: Frequency of transactions, location, etc.
Class Label: 1 for fraud, 0 for legitimate.

### Source:
The source of the project is payment_fraud.xls file that is uploaded.

### Approach
1. Data Preprocessing
Handling Missing Values: Cleaned the data to ensure no missing or inconsistent entries.
Encoding Categorical Variables: Converted any categorical variables into numerical formats using techniques like one-hot encoding.
Scaling the Data: Normalized transaction amounts and other continuous variables using standardization techniques like MinMaxScaler.
2. Feature Engineering
Time-Based Features: Extracted features such as the hour, day, and month of the transaction.
Aggregated Features: Created new features based on customer behavior, e.g., number of transactions in the past 24 hours, average transaction amount over a week.
Outlier Detection: Identified and flagged unusually large transactions.
Historical Features: Calculated averages or counts of customer transactions to detect deviations.
3. Model Building
Various machine learning algorithms were evaluated for fraud detection, including:
Machine learning
Neural Networks (optional)
4. Model Evaluation
Due to the class imbalance (fraud cases being rare), traditional accuracy was not the primary metric.
Key metrics:
Precision: Focused on reducing false positives.
Recall: Ensured a high detection rate for fraud (minimizing false negatives).
F1 Score: Balanced precision and recall.
ROC-AUC: Evaluated model discrimination ability.
Techniques like SMOTE (Synthetic Minority Over-sampling Technique) were used to balance the dataset.
### Installation
To run this project, you will need to have Python installed on your machine. Follow these steps to set up the environment:

Clone the repository:

bash
Copy code
```
git clone https://github.com/Reebak04/Payment-fraud-detection-using-machine-learning-and-feature-engineering.git
cd Payment-fraud-detection-using-machine-learning-and-feature-engineering
```
Install the required packages:

bash
Copy code
```
pip install -r requirements.txt
```

### Results
Best Performing Model: [Insert details of the best model, e.g., Random Forest with an F1 score of X].
The model was able to successfully detect fraudulent transactions with a high recall and acceptable precision.
### our project output :
![image](https://github.com/user-attachments/assets/96b605ba-d8b5-46f5-8135-50a3995b31af)
![image](https://github.com/user-attachments/assets/a3d1df27-de48-4751-8096-624eb2e477b1)
![image](https://github.com/user-attachments/assets/7f922c06-1aca-47c9-8eae-de374c70d533)


![image](https://github.com/user-attachments/assets/a72fe750-3e04-4c14-80d0-1aac388d89f7)

![image](https://github.com/user-attachments/assets/18500d71-c6ea-4439-ad84-7ee81fffdfd9)

### Conclusion :
From the above dataset we used there are 12,753 verified payments and 190 fraud payments.Thus we have achieved the goal of our project i.e Fraud Payment Detection with the accuracy of 1.0.




