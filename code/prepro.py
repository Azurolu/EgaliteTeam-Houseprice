#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: isabelleguyon

This is an example of program that preprocesses data.
It calls the PCA function from scikit-learn.
Replace it with programs that:
    normalize data (for instance subtract the mean and divide by the standard deviation of each column)
    construct features (for instance add new columns with products of pairs of features)
    select features (see many methods in scikit-learn)
    perform other types of dimensionality reductions than PCA
    remove outliers (examples far from the median or the mean; can only be done in training data)
"""

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA

class NothingTransformer:
    def fit(self, X, y):
        return X
    def fit_transform(self, X):
        return X
    def transform(self, X):
        return X
    def __str__(self) :
        return "Don't worry : Nothing's happening"

class Preprocessor(BaseEstimator):
    def __init__(self):
        '''
        This example does not have comments: you must add some.
        Add also some defensive programming code, like the (calculated) 
        dimensions of the transformed X matrix.
        '''
        self.transformer = NothingTransformer()
        print("PREPROCESSOR=" + self.transformer.__str__())

    def fit(self, X, y=None):
        print("PREPRO FIT")
        return self.transformer.fit(X, y)

    def fit_transform(self, X, y=None):
        print("PREPRO FIT_TRANSFORM")
        return self.transformer.fit_transform(X)

    def transform(self, X, y=None):
        print("PREPRO TRANSFORM")
        return self.transformer.transform(X)
    
if __name__=="__main__":
    # Put here your OWN test code
    
    # To make sure this runs on Codalab, put here things that will not be executed on Codalab
    from sys import argv, path
    path.append ("../ingestion") # Contains libraries you will need
    from data_manager import DataManager  # such as DataManager
    
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../../public_data" # Replace by correct path
        output_dir = "../results" # Create this directory if it does not exist
    else:
        input_dir = argv[1]
        output_dir = argv[2];
    
    basename = 'houseprice'
    D = DataManager(basename, input_dir) # Load data
    print("*** Original data ***")
    print D
    
    Prepro = Preprocessor()
 
    # Preprocess on the data and load it back into D
    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])
  
    # Here show something that proves that the preprocessing worked fine
    print("*** Transformed data ***")
    print D
    
