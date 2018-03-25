#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from sys import argv, path
path.append ("../../ingestion_program") # Contains libraries you will need
from data_manager import DataManager  # such as DataManager
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import Imputer
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.cluster import FeatureAgglomeration

from sklearn.decomposition import PCA

class Preprocessor(BaseEstimator):
    def __init__(self):
        self.transformer = PCA(n_components=2)

    def fit(self, X, y=None):
        return self.transformer.fit(X, y)

    def fit_transform(self, X, y=None):
        return self.transformer.fit_transform(X)

    def transform(self, X, y=None):
        return self.transformer.transform(X)
    
    def selectFeatures(self, X, y=None):
        clf = ExtraTreesClassifier()
        clf = clf.fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        return model
    
    def pipe(self, X, y=None):
         estimators = [('imputer',Imputer()),('scaler',StandardScaler()),
                       ('red_dim', LocallyLinearEmbedding()),
                       ('clustring',FeatureAgglomeration())]
         pipe = Pipeline(estimators)
         pipe.fit(X, y)
         return pipe

if __name__=="__main__":
    # We can use this to run this file as a script and test the Preprocessor
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../../sample_data"
        output_dir = "../../results" # Create this directory if it does not exist
    else:
        input_dir = argv[1]
        output_dir = argv[2];
    
    basename = 'Housing'
    D = DataManager(basename, input_dir) # Load data
    print("*** Original data ***")
    print D
    # Starting preprocessing
    Prepro = Preprocessor()
    # Defining variables
    X=np.copy(D.data['X_train'])
    y=np.copy(D.data['Y_train'])
    x_valid=np.copy(D.data['X_valid'])
    x_test=np.copy(D.data['X_valid'])
    # Selection of features and transformation
    model_selection = Prepro.selectFeatures(X, y)
    D.data['X_train'] = model_selection.transform(X)
    D.data['X_valid']=model_selection.transform(x_valid)
    D.data['X_test']=model_selection.transform(x_test)
    estimators = [('imputer',Imputer()),('scaler',MinMaxScaler()),('clustring',FeatureAgglomeration())]
    pipe = Pipeline(estimators)
    D.data['X_train']=pipe.fit_transform(D.data['X_train'],D.data['Y_train'])
    D.data['X_valid']=pipe.transform(D.data['X_valid'])
    D.data['X_test']=pipe.transform(D.data['X_test'])
  
    # Here show something that proves that the preprocessing worked fine
    print("*** Transformed data ***")
    print D
    
