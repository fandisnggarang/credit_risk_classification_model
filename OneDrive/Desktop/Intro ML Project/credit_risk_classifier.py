#!/usr/bin/env python
# coding: utf-8

# # TASK I: DATA PREPARATION

# In[268]:


import numpy as np
import pandas as pd
import copy


# ### Data Load

# In[269]:


def read_data(data):
  """
  Function to read data and clean duplicates and null values

  Parameters
  ---------
  data: DataFrame

  Returns
  ---------
  drop_dupl_na: DataFrame
    Data without duplicates and null values
  """
  data         = pd.read_csv(data)
  n_dupl       = data.duplicated().sum()
  n_Na         = data.isna().sum().sum()
  drop_dupl_na = data.drop_duplicates(keep='last').dropna()

  print(f'Data shape raw            : {data.shape}')
  print(f'Number of duplicate       : {n_dupl}')
  print(f'Number of na values       : {n_Na}')
  print(f'Data shape after dropping : {drop_dupl_na.shape}')

  return drop_dupl_na

data = read_data(data = 'credit_risk_dataset.csv')


# In[270]:


data.head()


# ### Data Splitting

# In[271]:


# input and output data splitting

def split_input_output(data, target_col):
  """
  Function to split data into input (X) and output (y)

  Parameters
  ---------
  data      : DataFrame
  target_col: str

  Returns
  ---------
  X, y: DataFrame
    X is the input and y is the output
  """
  X = data.drop(target_col, axis = 1)
  y = data[target_col]
  print('X shape :', X.shape)
  print('y shape :', y.shape)

  return X, y

X, y = split_input_output(data = copy.deepcopy(data), target_col = 'loan_status')


# In[272]:


X.head()


# In[273]:


y.head()
y.value_counts(normalize = True)


# In[274]:


# train, test (and valid) data splitting

from sklearn.model_selection import train_test_split

def split_train_test(X, y, test_size, seed):
  """
  Function to split data X and y into train and test data (which is then further split into valid data)

  Parameters
  ---------
  X, y     : DataFrame
  test_size: float
  seed     : int

  Returns
  ---------
  X_train, X_test, y_train, y_test: DataFrame
    X and y (train and test) data
  """
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = seed, stratify = y)
  print('X train shape:', X_train.shape)
  print('y train shape:', y_train.shape)
  print('X test shape :', X_test.shape)
  print('y test shape :', y_test.shape)
  print()

  return X_train, X_test, y_train, y_test



# In[275]:


# Splitting X and y into train and not_train data
X_train, X_not_train, y_train, y_not_train = split_train_test(X, y, test_size = 0.2, seed = 123)

# Separating not_train data into validation and test data
X_valid, X_test, y_valid, y_test           = split_train_test(X = X_not_train,
                                                              y = y_not_train,
                                                              test_size = 0.5,
                                                              seed = 123)


# In[276]:


print(len(X_train)/len(X))  # 0.8
print(len(X_valid)/len(X))  # 0.1
print(len(X_test)/len(X))   # 0.1


# ### Performing EDA

# In[216]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[277]:


def plotting_function(data, numerical, categorical):
    """
    Function to display numerical and categorical data in plot

    Parameters
    ---------
    data  : DataFrame
    hue   : str
    colors: str
 
    Returns
    ---------
    -
    """
    data_num_train = data.select_dtypes(exclude = ['object'])
    data_cat_train = data.select_dtypes(include = ['object'])
    if numerical: 
        fig, ax    = plt.subplots(nrows = 3, ncols = 3, figsize = (12, 8))
        axes       = ax.flatten()
        for i, col in enumerate(data_num_train.columns):
            sns.kdeplot(data_num_train[col], ax = axes[i])
            axes[i].set_title(f'Distribution of {col}')
    else:
        fig, axes  = plt.subplots(nrows = 1, ncols = len(data_cat_train.columns), figsize = (12, 6))
        for i, col in enumerate(data_cat_train.columns):
            data_cat_train[col].value_counts().plot(kind = 'bar', ax = axes[i])
            axes[i].set_title(col)
    plt.tight_layout()
    plt.show()
        


# In[278]:


# Calling the plotting_function to print the plot of numerical data
data_num_train = plotting_function(data = X_train, numerical = True, categorical = False)
data_num_train


# In[279]:


# Calling the plotting_function to print the plot of categorical data
data_cat_train = plotting_function(data = X_train, numerical = False, categorical = True)
data_cat_train


# In[223]:


# outlier checking
X_train.describe()


# In[280]:


X_train[X_train['person_age'] > 65].sort_values(by = 'person_age', ascending = False)


# 65 is the average retirement age worldwide
# Looking at data for customers above 65 years old


# In[281]:


X_train[X_train['person_emp_length'] > 45].sort_values(by = 'person_emp_length', ascending = False)

# 45 years is the average work tenure worldwide
# Looking at data for customers with work tenure above 45 years


# Comment:
# 1. The numerical data is skewed and the distribution of categorical data is imbalanced. Appropriate ML techniques are needed to address this.
# 2. There are outliers, which are also anomalous data points, namely person_age == 123 and 144 and person_emp_length == 123.
# 3. The categorical columns 'loan_grade' & 'cb_person_default_on_file' are numerically encoded with label encoding, the rest are encoded using One Hot.
# 
# Conclusion:
# 1. Removing outlier data based on assumptions.
# 2. Scaling the data to balance the data.
# 

# ### Data Preprocessing

# In[234]:


# removing the outliers
person_outlier     = X_train[(X_train['person_age'] > 65) | (X_train['person_emp_length'] > 45)].index
X_train_dropped    = X_train.drop(person_outlier)
y_train_dropped    = y_train.drop(person_outlier)

print('Shape of X train after dropped:', X_train_dropped.shape)
print('Shape of y train after dropped:', y_train_dropped.shape)


# In[227]:


X_train_dropped.describe()


# In[228]:


# Inserting categorical values with 'UNKNOWN'

from sklearn.impute import SimpleImputer


def cat_imputer_fit(data):
  """
  Function to perform fit imputation on data

  Parameters
  ---------
  data: DataFrame

  Returns
  imputer
     instance of SimpleImputer class
  """
  imputer     = SimpleImputer(strategy = 'constant', fill_value='UNKOWN')
  imputer     = imputer.fit(data)

  return imputer

def cat_imputer_transform(data, imputer):
  """
  Function to perform transform imputation on data

  Parameters
  ---------
  data   : DataFrame
  imputer: inscance

  Returns
  imputed_data
     DataFrame
  """
  imputing_data = imputer.transform(data)
  imputed_data  = pd.DataFrame(imputing_data)

  imputed_data.columns = data.columns
  imputed_data.index   = data.index

  return imputed_data

# Saving columns with categorical values to variable X_train_cat
X_train_cat = X_train_dropped.select_dtypes(['object'])

# Perform categorical imputation
cat_imputer = cat_imputer_fit(data = X_train_cat)

# Transform
X_train_dropped[X_train_cat.columns] = cat_imputer_transform(data = X_train_cat, imputer = cat_imputer)


# In[235]:


from sklearn.preprocessing import LabelEncoder

def apply_label_encoding(data, columns):
  """
  Function to perform label encoding

  Parameters
  ---------
  data   : DataFrame
  columns: list

  Returns
  data
     DataFrame
  """
  for col in columns:
      encoder = LabelEncoder()
      data[col+'_encoded'] = encoder.fit_transform(data[col])
  data.drop(columns, axis=1, inplace=True)
  return data

# Specify the columns to be encoded as labels
labels_encoding_col  = ['loan_grade', 'cb_person_default_on_file']

# Apply label encoding
X_train_dropped_encoded_label = apply_label_encoding(data = X_train_dropped, columns = labels_encoding_col)


# In[236]:


X_train_dropped_encoded_label.head(5)


# In[237]:


# Check the result of label encoding
X_train_dropped_encoded_label[['loan_grade_encoded']].value_counts()


# In[238]:


# Check the result of label encoding
X_train_dropped_encoded_label[['cb_person_default_on_file_encoded']].value_counts()


# In[239]:


from sklearn.preprocessing import OneHotEncoder

def apply_oneHot_encoding(data, columns):
  """
  Function to perform OneHot encoding

  Parameters
  ---------
  data   : DataFrame
  columns: list

  Returns
  data
     DataFrame
  """
  for col in columns:
      encoder = OneHotEncoder()
      encoded_cols = encoder.fit_transform(data[[col]])
      feature_names = encoder.get_feature_names_out([col])
      encoded_data = pd.DataFrame(encoded_cols.toarray(), columns=feature_names, index=data.index)
      data = pd.concat([data, encoded_data], axis=1)
      data.drop(col, axis=1, inplace=True)
  return data

# Specify the columns to be one-hot encoded
oneHot_encoding_col = ['person_home_ownership', 'loan_intent']

# Apply one-hot encoding
X_train_dropped_encoded_Ohe = apply_oneHot_encoding(data=X_train_dropped_encoded_label, columns=oneHot_encoding_col)


# In[240]:


X_train_dropped_encoded_Ohe.head()


# In[241]:


# Scaling numerical values

from sklearn.preprocessing import StandardScaler

def fit_scaler(data):
  """
  Function to perform fit scaling on data

  Parameters
  ---------
  data: DataFrame

  Returns
  standardizer
     transformer
  """
  standardizer = StandardScaler()
  standardizer.fit(data)
  return standardizer

def transform_scaler(data, scaler):
  """
  Function to perform transform scaling on data

  Parameters
  ---------
  data: DataFrame

  Returns
  scaler
     object
  """
  scaled_data_raw = scaler.transform(data)
  scaled_data     = pd.DataFrame(scaled_data_raw)

  scaled_data.columns = data.columns
  scaled_data.index   = data.index

  return scaled_data

# Fit scaler
scaler = fit_scaler(data = X_train_dropped_encoded_Ohe)

# Transform scaler
X_train_clean = transform_scaler(data = X_train_dropped_encoded_Ohe, scaler = scaler)


# In[242]:


X_train_clean.describe()


# In[243]:


X_train_clean.shape


# ### Preprocessing X_valid and X_test

# In[244]:


def preprocessing_data(data, label_cols, ohe_cols):
  """
  Function to perform preprocessing including categorical imputation, label encoding, OneHot encoding, and scaling

  Parameters
  ---------
  data      : DataFrame
  label_cols: list
  ohe_cols  : list

  Returns
  data_clean
     DataFrame
  """
  # IMPUTATION
  data_cat = data.select_dtypes(['object'])

  # Perform categorical imputation
  cat_imputer = cat_imputer_fit(data_cat)

  # Transform
  data[data_cat.columns] = cat_imputer_transform(data_cat, cat_imputer)

  # Apply label encoding
  data_encoded_label = apply_label_encoding(data, label_cols)

  # Apply one-hot encoding
  data_encoded_Ohe = apply_oneHot_encoding(data_encoded_label, ohe_cols)

  # SCALING
  # Fit scaler
  scaler = fit_scaler(data = data_encoded_Ohe)

  # Transform scaler
  data_clean = transform_scaler(data = data_encoded_Ohe, scaler = scaler)

  return data_clean


# In[245]:


# preprocessing data X_valid
X_valid_clean = preprocessing_data(X_valid, labels_encoding_col, oneHot_encoding_col)


# In[246]:


X_valid_clean.describe()


# In[247]:


# preprocessing data X_test
X_test_clean  = preprocessing_data(X_test, labels_encoding_col, oneHot_encoding_col)


# In[248]:


X_test_clean.describe()


# # TASK II: MODELING

# In[249]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[250]:


def extract_cv_results(cv_obj):
  """
  Function to extract scores data after cross-validation process

  Parameters
  ---------
  cv_obj: object

  Returns
  train_score, valid_score, best_params
     numerik, numerik, str
  """
  train_score = cv_obj.cv_results_['mean_train_score'][cv_obj.best_index_]
  valid_score = cv_obj.cv_results_['mean_test_score'][cv_obj.best_index_]
  best_params = cv_obj.best_params_

  return train_score, valid_score, best_params


# Comment:
# 
# The selected classification algorithms are as follows: DummyClassifier as the baseline, followed by Logistic Regression, Decision Tree, and Random Forest. The chosen evaluation metric is 'roc_auc'. This metric is selected because it can address the business case in credit risk analysis, which seems to emphasize anticipating the emergence of false positive applicants.

# ### Baseline Model

# In[251]:


# create dummy classifier
dummy_clf = DummyClassifier(strategy="most_frequent")

param_grid_baseline = {}
reg_base = GridSearchCV(estimator=dummy_clf, param_grid=param_grid_baseline, cv=10, scoring='roc_auc', return_train_score=True)

reg_base.fit(X_train_clean, y_train_dropped)

# extraction of the cross-validation process and printing the results
train_reg_base, valid_reg_base, best_param_reg_base = extract_cv_results(reg_base)
print(f'Train score - Logistic Regression: {train_reg_base}')
print(f'Valid score - Logistic Regression: {valid_reg_base}')
print(f'Best Params - Logistic Regression: {best_param_reg_base}')


# ### Logistic Regression

# Comment:
# 
# The following parameters are used because:
# 1. 'penalty': 
# L1 (Lasso) serves as a method for feature selection, and L2 (Ridge) is used to help prevent overfitting.
# 
# 2. 'C': 
# This is the regularization strength, where smaller values are stronger and help prevent overfitting. Larger values allow the model to pay more attention to the training data.
# 
# 3. solver: 
# The 'liblinear' solver is suitable for datasets that are relatively small to medium in size. Additionally, this solver is suitable for handling both L1 (Lasso) and L2 (Ridge) regularization.
# 
# 4. class_weight: 
# This parameter ('balanced') provides balanced weights for each class in logistic regression. It is used to handle imbalanced data in y.

# In[252]:


# defining the parameter
param_grid_logistic = {
    'penalty': ['l1', 'l2'],
    'C'      : [0.001, 0.01, 0.1, 1, 10]
}

# defining a LogisticRegression model with solver 'liblinear' and class_weight 'balanced'
logistic_model = LogisticRegression(
                                    solver       = 'liblinear',
                                    class_weight = 'balanced',
                                    random_state = 123)

# initializing GridSearchCV
reg_logistic = GridSearchCV(estimator=logistic_model, param_grid=param_grid_logistic, cv=10, scoring= 'roc_auc', return_train_score=True)

# fit model
reg_logistic.fit(X_train_clean, y_train_dropped)

# extraction of the cross-validation process and printing the results
train_logistic, valid_logistic, best_param_logistic = extract_cv_results(reg_logistic)
print(f'Train score - Logistic Regression: {train_logistic}')
print(f'Valid score - Logistic Regression: {valid_logistic}')
print(f'Best Params - Logistic Regression: {best_param_logistic}')


# ### Decision Tree

# Comment:
# 
# 1. 'max_depth': 
# This parameter limits the maximum depth of the decision tree. By limiting the depth, we can control the complexity of the model and prevent overfitting. Choosing a lower value can result in a more interpretable model but potentially less accuracy.
# 
# 2. 'min_samples_split': 
# This parameter determines the minimum number of samples required to split a node in the tree. By increasing this value, we direct the model to make more significant splits in the data, which can help reduce overfitting.
# 
# 3. 'min_samples_leaf': 
# This parameter determines the minimum number of samples required to be a leaf node in the tree. By increasing this value, we prevent the model from creating very small partitions, which can help reduce overfitting and make the model more general.

# In[254]:


# defining the parameter
param_grid_tree = {
    'max_depth'        : [4, 6],
    'min_samples_split': [10, 15],
    'min_samples_leaf' : [2, 4]
}

# defining DecisionTreeClassifier
tree_model = DecisionTreeClassifier(class_weight='balanced')

# initializing GridSearchCV
reg_tree = GridSearchCV(estimator=tree_model, param_grid=param_grid_tree, cv=10, scoring= 'roc_auc', return_train_score=True)

# Fit model
reg_tree.fit(X_train_clean, y_train_dropped)

# extraction of the cross-validation process and printing the results
train_tree, valid_tree, best_param_tree = extract_cv_results(reg_tree)
print(f'Train score - DecisionTreeClassifier: {train_tree}')
print(f'Valid score - DecisionTreeClassifier: {valid_tree}')
print(f'Best Params - DecisionTreeClassifier: {best_param_tree}')



# ### Random Forest

# Comment:
# 
# 1. 'n_estimators': 
# The number of trees in the forest. The more trees, the better the generalization usually, but the longer it takes to train the model.
# 
# 2. 'max_depth': 
# The maximum depth of each tree in the forest.

# In[255]:


# specify the range of values for n_estimators
B = [10, 15, 20, 25, 30, 35, 40]

# defining the parameter
param_grid_random = {"n_estimators": B, "max_depth": [2, 4, 6, 8]}

# defining RandomForestClassifier
rf_model = RandomForestClassifier(class_weight='balanced', random_state=123)

# initializing GridSearchCV
rf_cv   = GridSearchCV(estimator=rf_model, param_grid=param_grid_random, cv=10, scoring= 'roc_auc', return_train_score=True)

# Fit model
rf_cv.fit(X_train_clean, y_train_dropped)

# extraction of the cross-validation process and printing the results
train_rf, valid_rf, best_param_rf = extract_cv_results(rf_cv)
print(f'Train score auc - RandomForestClassifier: {train_rf}')
print(f'Valid score auc - RandomForestClassifier: {valid_rf}')
print(f'Best Params     - RandomForestClassifier: {best_param_rf}')


# # TASK III: MODEL EVALUATION

# In[256]:


# Summary of modeling results (model name, scoring type, and best parameters)
summary_df = pd.DataFrame(
    data={
        'model'      : ['Dummy Classifier', 'Logistic Regression', 'Decision Tree', 'Random Forest'],
        'train_score': [train_reg_base, train_logistic, train_tree, train_rf],
        'valid_score': [valid_reg_base, valid_logistic, valid_tree, valid_rf],
        'best_params': [best_param_reg_base, best_param_logistic, best_param_tree, best_param_rf]
    }
)


summary_df['train_score'] /= 10**6
summary_df['valid_score'] /= 10**6
summary_df


# Comment:
# 
# The best model selected is Random Forest, as it outperformed the Dummy Classifier, Logistic Regression, and Decision Tree in both training and validation scores.
# 
# The CV results show that the Random Forest model with the best parameters (max_depth=8, n_estimators=40) achieved an AUC score of around 0.9375 on the training data and 0.9219 on the validation data. These scores indicate good performance and suggest that the model can generally generalize well to unseen data.
# 
# Next, as an initial step in evaluating the model, a function is built to find the threshold and cost function. The goal is to see at which threshold the class division occurs and the cost function generated from false positive and false negative predictions, as emphasized in the question.
# 
# The question specifies the following:
# - if you falsely predict good applicants as bad, you would lose potential revenue of Rp 10.000.000/applicant on average (I conclude that this refers to False Positive (good aplicants = 0, default/bad = 1))
# 
# - if you falsely predict bad applicants as good, you would lose Rp 35.000.000/applicant on average (I conclude that this refers to False Negative (good aplicants = 0, default/bad = 1))
# 
# The business case seems to emphasize false positives and false negatives. However, upon comparison, it appears that the business is more concerned with avoiding false negatives than false positives. This is evident from the potential loss of false negatives, which is higher at Rp 35,000,000/applicant compared to Rp 10,000,000/applicant for false positives.

# ### Threshold and Cost Function

# In[257]:


def threshCost_function(cost_fp, cost_fn, thresholds, actual, pred_proba):

  """ 
  Function to find the class division threshold in the model and the resulting cost function

  Parameters
  ---------
  cost_fp   : int
  cost_fn   : int
  thresholds: float
  actual    : DataFrame
  pred_proba: DataFrame

  Returns
  best_threshold
    float
  min_cost
     int
  """
  # Initializing variables to store the best threshold and minimum cost
  best_threshold = None
  min_cost = float('inf')

  # iterating over threshold values
  for threshold in thresholds:
    # Converting prediction probabilities to binary predictions based on the threshold
    y_train_pred_binary = (pred_proba >= threshold).astype(int)

    # summing the number of false positives and false negatives
    fp = np.sum((actual == 0) & (y_train_pred_binary == 1))
    fn = np.sum((actual == 1) & (y_train_pred_binary == 0))

    # calculating the total_cost based on the threshold
    total_cost = cost_fp * fp + cost_fn * fn

    # updating the best threshold and minimum cost if the threshold has a minimum cost
    if total_cost < min_cost:
        min_cost = total_cost
        best_threshold = threshold

  # Printing the best threshold and minimum cost
  print(f'Best Threshold    : {best_threshold}')
  print(f'Minimum Cost      : {min_cost}')

  return best_threshold, min_cost


# In[258]:


best_model = RandomForestClassifier(class_weight='balanced',
                                      max_depth   = rf_cv.best_params_['max_depth'],
                                      n_estimators= rf_cv.best_params_['n_estimators'],
                                      random_state=123)

best_model.fit(X_train_clean, y_train_dropped)


# In[259]:


# predicting probabilities on the training data
y_train_pred_proba = best_model.predict_proba(X_train_clean)[:, 1]

# define the range of threshold values
thresholds = np.linspace(0, 1, 100)

# define the total cost of false positives and false negatives
cost_fp = 10000000  # Cost false positives
cost_fn = 35000000  # Cost false negatives

threshold_and_cost = threshCost_function(cost_fp = cost_fp, cost_fn = cost_fn, thresholds = thresholds, actual = y_train_dropped, pred_proba = y_train_pred_proba)


# In[260]:


# predicting probabilities on the valid data
y_valid_pred_proba = best_model.predict_proba(X_valid_clean)[:, 1]

# define the range of threshold values
thresholds = np.linspace(0, 1, 100)

# define the total cost of false positives and false negatives
cost_fp = 10000000  # Cost false positives
cost_fn = 35000000  # Cost false negatives

threshold_and_cost = threshCost_function(cost_fp = cost_fp, cost_fn = cost_fn, thresholds = thresholds, actual = y_valid, pred_proba = y_valid_pred_proba)


# In[261]:


# predicting probabilities on the test data
y_test_pred_proba = best_model.predict_proba(X_test_clean)[:, 1]

# define the range of threshold values
thresholds = np.linspace(0, 1, 100)

# define the total cost of false positives and false negatives
cost_fp = 10000000  # Cost false positives
cost_fn = 35000000  # Cost false negatives

threshold_and_cost = threshCost_function(cost_fp = cost_fp, cost_fn = cost_fn, thresholds = thresholds, actual = y_test, pred_proba = y_test_pred_proba)


# Comment:
# It can be seen that the best threshold for the training, validation, and test data is in the range of 0.36 - 0.49, with cost functions ranging from Rp47,720,000,000 to Rp7,495,000,000.

# ### Model Prediction: Roc Auc Score & Confusion Matrix
# 

# In[262]:


# predicting & evaluating the training data
y_train_pred= best_model.predict(X_train_clean)

rf_f1_train = roc_auc_score(y_train_dropped, y_train_pred)

print(f'Score on data train : {rf_f1_train}')
print()

# show the confusion matrix results
print('Confusion Matrix    : ')
confusion_matrix(y_train_dropped, y_train_pred)


# In[263]:


# checking the cost of FP and FN, comparing it with the cost generated by the threshCost_function (47720000000)
fp           = 880  * 10000000
fn           = 1115 * 35000000
print(fp+fn)


# In[264]:


# predicting & evaluating the valid data
y_valid_pred= best_model.predict(X_valid_clean)

rf_f1_valid = roc_auc_score(y_valid, y_valid_pred)

print(f'Score on data valid: {rf_f1_valid}')
print()

# show the confusion matrix results
print('Confusion Matrix   : ')
confusion_matrix(y_valid, y_valid_pred)


# In[265]:


# checking the cost of FP and FN, comparing it with the cost generated by the threshCost_function (6580000000)
fp           = 139 * 10000000
fn           = 155 * 35000000
print(fp+fn)


# In[266]:


# predicting & evaluating the valid data
y_test_pred = best_model.predict(X_test_clean)

rf_f1_test  = roc_auc_score(y_test, y_test_pred)

print(f'Score on data test : {rf_f1_test}')
print()

# show the confusion matrix results
print('Confusion Matrix   : ')
confusion_matrix(y_test, y_test_pred)


# In[267]:


# checking the cost of FP and FN, comparing it with the cost generated by the threshCost_function (7495000000)
fp           = 94  * 10000000
fn           = 192 * 35000000
print(fp+fn)


# Comment:
# 
# The prediction scores on the training, validation, and test data are 0.8626, 0.8436, and 0.8239, respectively. These scores indicate that the model performs quite well on all three datasets, with slightly lower performance on the test set, as expected.
# 
# From the dataset, with a score of 0.82, here are the confusion matrix results:
# - True positives  (TP) : 427
# - False positives (FP) : 94
# - True negatives  (TN) : 2138
# - False negatives (FN) : 192
# 
# Regarding the business case, our focus is on false positives and false negatives, with explanations for each category as follows:
# 
# - False positives (FP) : 
# The model incorrectly predicted 94 negative cases as positive, resulting in a potential loss of Rp940,000,000 (94*10,000,000)
# 
# - False negatives (FN) : 
# The model incorrectly predicted 192 positive cases as negative, resulting in a potential loss of Rp67,200,000,000 (192*35,000,000)
# 
# Certainly, in the future, the performance of the model with a score of 0.82 on the dataset needs to be further optimized so that the number of incorrect predictions for FP and FN continues to decrease.
