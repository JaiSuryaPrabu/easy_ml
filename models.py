import pandas as pd

# Regression models
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# for multiprocessing
from concurrent.futures import ThreadPoolExecutor

class regression:
    '''
    regression is a class that uses multiprocessing method to train all the regression model and produce the accuracy as result
    # Arguments:
    
    X_train : Numpy array of independent variable that the model needs to be trained
    
    y_train : Numpy array of dependent variable that the model needs to be trained
    
    X_test : Numpy array of dependent variable that the model needs to be evaluated
    
    y_test : Numpy array of dependent variable that the model needs to be evaluated

    # Code
    ``` python
    >>> r = regression(X_train,y_train,X_test,y_test)
    >>> r.result()
    {'Model Name': [...], 'Accuracy': [...]}
    >>> r.get_best()
    LinearRegression()
    ```
    '''
    def __init__(self,X_train,y_train,X_test,y_test,verbose=False):
        self.lin_reg = LinearRegression()
        self.ridge = Ridge()
        self.lasso = Lasso()
        self.elastic = ElasticNet()
        self.svr = LinearSVR()
        self.decision_tree = DecisionTreeRegressor()
        self.random_forest = RandomForestRegressor()
        
        self.num_of_models = 7
        self.verbose = verbose
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.meta_data = {
            "Model Name":["Linear Regression","Ridge","Lasso","Elastic Net","SVR","Decision Tree","Random Forest"],
            "Accuracy":[],
            "Model":[self.lin_reg,self.ridge,self.lasso,self.elastic,self.svr,self.decision_tree,self.random_forest]
        }

        self.train()
    
    def train_linear(self):
        if self.verbose:
            print("\tTraining Linear Regression")
        
        self.lin_reg.fit(self.X_train,self.y_train)
        
        accuracy = r2_score(y_true=self.y_test,y_pred=self.lin_reg.predict(self.X_test))
        return accuracy
        
    def train_ridge(self):
        if self.verbose:
            print("\tTraining Ridge")
        
        self.ridge.fit(self.X_train,self.y_train)
        
        accuracy = r2_score(y_true=self.y_test,y_pred=self.ridge.predict(self.X_test))
        return accuracy
            
    def train_lasso(self):
        if self.verbose:
            print("\tTraining Lasso")
        
        self.lasso.fit(self.X_train,self.y_train)
        
        accuracy = r2_score(y_true=self.y_test,y_pred=self.lasso.predict(self.X_test))
        return accuracy
    
    def train_elastic(self):
        if self.verbose:
            print("\tTraining Elastic")
        
        self.elastic.fit(self.X_train,self.y_train)
        
        accuracy = r2_score(y_true=self.y_test,y_pred=self.elastic.predict(self.X_test))
        return accuracy
    
    def train_svr(self):
        if self.verbose:
            print("\tTraining Support Vector Regression")
        self.svr.fit(self.X_train,self.y_train)
        
        accuracy = r2_score(y_true=self.y_test,y_pred=self.svr.predict(self.X_test))
        return accuracy
    
    def train_tree(self):
        if self.verbose:
            print("\tTraining Decision Tree")
        self.decision_tree.fit(self.X_train,self.y_train)
        
        accuracy = r2_score(y_true=self.y_test,y_pred=self.decision_tree.predict(self.X_test))
        return accuracy
    
    def train_forest(self):
        if self.verbose:
            print("\tTraining Random Forest")
        self.random_forest.fit(self.X_train,self.y_train)
        
        accuracy = r2_score(y_true=self.y_test,y_pred=self.random_forest.predict(self.X_test))
        return accuracy
    
    def train(self):
        '''train the regression model with the multiprocessing'''
        if self.verbose:
            print("Training begins ...")
        with ThreadPoolExecutor(max_workers=self.num_of_models) as executor:
            f1 = executor.submit(self.train_linear)
            f2 = executor.submit(self.train_ridge)
            f3 = executor.submit(self.train_lasso)
            f4 = executor.submit(self.train_elastic)
            f5 = executor.submit(self.train_svr)
            f6 = executor.submit(self.train_tree)
            f7 = executor.submit(self.train_forest)
            
            self.meta_data["Accuracy"] = [f1.result(),f2.result(),f3.result(),f4.result(),f5.result(),f6.result(),f7.result()]
        
        if self.verbose:
            print("Training Completed ...")
    
    def result(self):
        return pd.DataFrame({
            "Model"   : self.meta_data["Model Name"],
            "Accuracy": self.meta_data["Accuracy"]
        })
    
    def get_best(self):
        max_acc = max(self.meta_data["Accuracy"])
        index = self.meta_data["Accuracy"].index(max_acc)

        if self.verbose:
            print("The best model is ",self.meta_data["Model Name"][index])
            
        best_model = self.meta_data["Model"][index]
        return best_model