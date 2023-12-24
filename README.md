# EASY ML
**easy_ml** is a machine learning library that executes on top of the **scikit-learn**.
This is a small project where you can train multiple models in a single line and get the best perfoming model.
**easy_ml** is under development and this project is open to contributions from anyone 🤗 

# Roadmap
## 1. Regression
1. Linear Models:
    1. Linear Regression - For simple and multiple linear regression
    2. Ridge
    3. Lasso
    4. ElasticNet
2. Support Vector Machine
    1. LinearSVR
3. Decision Tree
    1. DecisionTreeRegressor
4. Ensemble
    1. RandomForestRegressor
## 2. Classification
~ On the way ~
## 3. Clustering
~ On the way ~
## 4. Data Preprocessing
~ On the way ~
## 5. Upload to PyPi
~ On the way ~

## Metrics
1. R2 Score for regression

# Sample Code
```python
from easy_ml.models import regression
reg = regression(X_train,y_train,X_train,y_test)
result = reg.result() # returns the pandas dataframe
model = reg.get_best() 
```