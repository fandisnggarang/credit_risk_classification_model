## Credit Risk Classification Model

This is a simple project, credit risk classification model, that is part of the intro to Machine Learning exercises in Pacmann. The goal of this project is to understand the basic workflow of creating a Machine Learning model, as well as applying analysis in an industry case, using 4 simple algorithms.

We are encouraged to work as data scientists in a risk analysis team in the finance industry. Our company profits by providing loans to customers but faces potential losses if customers fail to repay the loan (known as defaulting). To minimize these losses, it is important to prevent bad applicants (who may later default) from receiving loans. Our target variable is loan status which consists of binary identification: 0 is non default, 1 is default. 

As data scientists wanna be, our goal is to build a classifier model that can accurately classify applicants as good or bad. If we incorrectly classify a good applicant as bad, we would lose an average potential revenue of Rp10,000,000 per applicant. On the other hand, if we incorrectly classify a bad applicant as good, we would lose Rp35,000,000 per applicant on average.

Comprehensive overview of the Credit Risk dataset: 

<img width="500" alt="Screenshot 2024-04-05 231546" src="https://github.com/fandisnggarang/credit_risk_classifier/assets/141505705/e2049b80-0106-4e83-b794-f8ea355f139a">

Source of picture and dataset: [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

Regarding the business case provided by Pacmann team, our focus is on false positives and false negatives. These are the results of the best model:

- False positives (FP) : 
The model incorrectly predicted 94 negative cases as positive, resulting in a potential loss of Rp940,000,000 (94*10,000,000)

- False negatives (FN) : 
The model incorrectly predicted 192 positive cases as negative, resulting in a potential loss of Rp67,200,000,000 (192*35,000,000)

Certainly, in the future, the performance of the model with a score of 0.82 on the dataset needs to be further optimized so that the number of incorrect predictions for FP and FN continues to decrease.

