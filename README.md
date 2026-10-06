# Telco Customer Churn Prediction

I built a model that predicts which telecom customers are likely to leave, using the IBM Telco Customer Churn dataset (7,043 customers, 21 columns, about 26.5% of them churned). The goal is to flag at-risk customers early so a company could offer them a retention deal before they go.

Built with Python, pandas, scikit-learn and XGBoost in Google Colab.

## What I did
- Cleaned the data: 11 blank `TotalCharges` values were converted to missing and filled with the median.
- Split into train and test sets (80/20, stratified) before any preprocessing, and kept all scaling and encoding inside a scikit-learn pipeline so nothing from the test set leaks into training.
- Compared 6 models using 5-fold stratified cross-validation (ROC-AUC) on the training data only.
- Tuned the decision threshold using out-of-fold predictions on the training set, then tested once on the held-out 20%.

## Model comparison (5-fold CV ROC-AUC)

| Model | ROC-AUC |
|---|---|
| Gradient Boosting | 0.848 |
| Logistic Regression | 0.846 |
| XGBoost | 0.845 |
| KNN | 0.833 |
| Random Forest | 0.821 |
| Extra Trees | 0.788 |

The top three are within noise of each other (standard deviation about 0.012), so a simple Logistic Regression would be a fair choice if explainability matters more than a tiny score gain. I used Gradient Boosting because it had the highest average.

## Results on the held-out test set
- Test ROC-AUC: 0.843
- At the default 0.5 threshold: accuracy 0.81, churn recall 0.52, churn precision 0.67
- At the tuned 0.35 threshold: accuracy 0.78, churn recall 0.71, churn precision 0.56

Always predicting "no churn" would already give about 73.5% accuracy, so accuracy alone says very little here. I focused on recall and precision for the customers who actually churn.

## Why I lowered the threshold
Missing a customer who is about to leave usually costs more than sending a retention offer to someone who would have stayed. Moving the threshold from 0.5 to 0.35 let the model catch about 265 of the 374 churners in the test set, instead of about 195. The trade-off is more false alarms and slightly lower accuracy. The right threshold in real life depends on what a retention offer actually costs, which I don't know.

## What drives churn
Feature importance from the Gradient Boosting model shows these as the strongest signals:
- Month-to-month contracts (by far the largest)
- Short tenure
- Fiber optic internet service
- Higher total and monthly charges
- No online security or tech support, and paying by electronic check

In short, customers who are new, on flexible contracts and paying more are the highest risk. This shows what the model relies on, not what causes churn. Tenure and total charges are also related, so their separate importance is hard to untangle.

## Limitations
- One dataset, with no external validation.
- The test set has 1,409 customers, so small differences between models are not reliable.
- The model is not tuned beyond the threshold. Hyperparameter search could add a little.

## How to run
1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle (IBM sample dataset).
2. Upload it to Google Colab.
3. Run the notebook from top to bottom.

## Author
Rafia, Electronics Engineering student at MUET, working on machine learning and data analysis.
