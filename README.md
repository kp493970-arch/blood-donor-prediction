# Blood Donor Return Prediction

Logistic regression model predicting whether a blood donor will donate again.
Dataset: UCI Blood Transfusion Service Center (748 real donors).

## Problem
Blood banks need to identify donors likely to return for targeted reminder 
campaigns. Only 24% of donors returned - a class imbalance problem where 
accuracy is a misleading metric.

## My Approach
- Dropped Monetary feature (= Frequency x 250, multicollinear - no new info)
- Stratified 75/25 train/test split to preserve class ratio
- StandardScaler fit on training data only - avoids data leakage
- Logistic Regression with balanced class weights - handles class imbalance

## Results
| Metric | Value |
|--------|-------|
| ROC-AUC | 0.785 |
| Donor Recall | 84% |
| Donors caught (test set) | 37 out of 44 |

## Key Finding
Recency is the strongest predictor - donors who gave blood recently 
are significantly more likely to return. Makes clinical sense.

## What I'd Do Next
- k-fold cross-validation for more stable AUC estimate
- Compare against Random Forest
- SMOTE oversampling as alternative to class weighting

## Stack
Python · pandas · scikit-learn · matplotlib

## Author
Kunal Patil - MSc Digital Health, Deggendorf Institute of Technology
