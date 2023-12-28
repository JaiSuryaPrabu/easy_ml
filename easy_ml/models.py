import pandas as pd

# Regression models
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB

from sklearn.metrics import r2_score,accuracy_score

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
    >>> r = regression(X_train,y_train,X_test,y_test,False)
    >>> r.result()
        Model Name  Accuracy
    0   Linear Regression   0.9
    ...
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
            print("Training Begins ...")
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

class classification:
    '''
    classification is a class that uses muliprocessing method to train all the classification models
    # Arguments:
    
    X_train : Numpy array of independent variable that the model needs to be trained
    
    y_train : Numpy array of dependent variable that the model needs to be trained
    
    X_test : Numpy array of dependent variable that the model needs to be evaluated
    
    y_test : Numpy array of dependent variable that the model needs to be evaluated

    # Code
    ``` python
    >>> c = classification(X_train,y_train,X_test,y_test,False)
    >>> c.result()
        Model Name  Accuracy
    0   Logistic Regression 0.5
    ...
    >>> c.get_best()
    LogisticRegression()
    '''

    def __init__(self,X_train,y_train,X_test,y_test,verbose=False):
        self.log_reg = LogisticRegression()
        self.svc = SVC()
        self.tree = DecisionTreeClassifier()
        self.forest = RandomForestClassifier()
        self.neighbors = KNeighborsClassifier()
        self.gauss = GaussianNB()
        self.bernoulli = BernoulliNB()
        self.multinomial = MultinomialNB()

        self.num_of_models = 8
        self.verbose = verbose

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.meta_data = {
            "Model Name":["Logistic Regression","SVC","Decision Tree","Random Forest","K Neighbors","Gaussian","Bernoulli","Multinomial"],
            "Accuracy":[],
            "Model":[self.log_reg,self.svc,self.tree,self.forest,self.neighbors,self.gauss,self.bernoulli,self.multinomial]
        }

        self.train()

    def train_linear(self):
        if self.verbose:
            print("\t Training Logistic Regression")
        self.log_reg.fit(self.X_train,self.y_train)
        acc = accuracy_score(self.y_test,self.log_reg.predict(self.X_test))
        return acc
    
    def train_svc(self):
        if self.verbose:
            print("\t Training SVC")
        self.svc.fit(self.X_train,self.y_train)
        acc = accuracy_score(self.y_test,self.svc.predict(self.X_test))
        return acc
    
    def train_tree(self):
        if self.verbose:
            print("\t Training Decision Tree")
        self.tree.fit(self.X_train,self.y_train)
        acc = accuracy_score(self.y_test,self.tree.predict(self.X_test))
        return acc
    
    def train_forest(self):
        if self.verbose:
            print("\t Training Random Forest")
        self.forest.fit(self.X_train,self.y_train)
        acc = accuracy_score(self.y_test,self.forest.predict(self.X_test))
        return acc
    
    def train_neighbors(self):
        if self.verbose:
            print("\t Training K Neighbors Classifier")
        self.neighbors.fit(self.X_train,self.y_train)
        acc = accuracy_score(self.y_test,self.neighbors.predict(self.X_test))
        return acc
    
    def train_gauss(self):
        if self.verbose:
            print("\t Training Gaussian Naive Bayes")
        self.gauss.fit(self.X_train,self.y_train)
        acc = accuracy_score(self.y_test,self.gauss.predict(self.X_test))
        return acc
    
    def train_bernoulli(self):
        if self.verbose:
            print("\t Training Bernoulli Naive Bayes")
        self.bernoulli.fit(self.X_train,self.y_train)
        acc = accuracy_score(self.y_test,self.bernoulli.predict(self.X_test))
        return acc
    
    def train_multinomial(self):
        if self.verbose:
            print("\t Training Multinomial Naive Bayes")
        self.multinomial.fit(self.X_train,self.y_train)
        acc = accuracy_score(self.y_test,self.multinomial.predict(self.X_test))
        return acc
    
    def train(self):
        '''training all the classification models with the multiprocessing'''
        if self.verbose:
            print("Training Begins ...")
        
        with ThreadPoolExecutor(max_workers=self.num_of_models) as exe:
            m1 = exe.submit(self.train_linear)
            m2 = exe.submit(self.train_svc)
            m3 = exe.submit(self.train_tree)
            m4 = exe.submit(self.train_forest)
            m5 = exe.submit(self.train_neighbors)
            m6 = exe.submit(self.train_gauss)
            m7 = exe.submit(self.train_bernoulli)
            m8 = exe.submit(self.train_multinomial)

            self.meta_data["Accuracy"] = [m1.result(),m2.result(),m3.result(),m4.result(),m5.result(),m6.result(),m7.result(),m8.result()]

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