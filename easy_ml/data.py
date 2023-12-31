''' Types of datasets
1. Structured - tabular , time series , spatial (geopandas)
2. Unstructured - text , image , audio , video
'''
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split

class TabularData:
    '''
    TabularData is a class that is used for data preprocessing in the tabular data.
    This class provides data with feature scaled and not feature scaled to better understand the ml model
    '''
    def __init__(self,path,to_summarize=False,verbose=False):
        
        self.dataset = pd.read_csv(path)
        self.missing_values()
        self.X = self.dataset.iloc[:,:-1].values
        self.y = self.dataset.iloc[:,-1].values
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.verbose = verbose

        
        self.encode_categorical_data()
        self.split_dataset()
        self.feature_scale()
        
        if to_summarize:
            self.summary()

    def summary(self):
        '''summary() provides the statisticall summary of the data
            It returns : head(),info(),describe()
        '''
        pass

    def missing_values(self):
        '''misssing_values() handles when there is missing values in the data'''
        
        # Finding the missing values
        data = self.dataset.copy()
        cols_with_missing_values = data.columns[data.isnull().any()]

        imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
        imputer = imputer.fit(data[cols_with_missing_values])
        data[cols_with_missing_values] = imputer.transform(data[cols_with_missing_values])

        self.dataset = data

    def encode_categorical_data(self):
        '''encode_categorical_data() is used to convert the text in a cell to numbers'''
        array = [] # find the categorical value array
        ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),array)],remainder='passthrough')
        self.X = np.array(ct.fit_transform(self.X))

        # if the dependent variable y contains the categorical value?
        '''
        le = LabelEncoder()
        self.y = le.fit_transform(y)
        '''

    def split_dataset(self,test_size):
        '''split the data into ratios of ...'''
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=test_size)

    def feature_scale(self):
        '''feature_scale*() is used to scale all the data to same scale'''
        sc = StandardScaler()
        # column from and to is need to be found
        column_from = 0
        column_to = 0
        self.X_train[:,column_from:column_to] = sc.fit_transform(self.X_train[:,column_from:column_to])
        self.X_test[:,column_from:column_to] = sc.fit_transform(self.X_test[:,column_from:column_to])
        