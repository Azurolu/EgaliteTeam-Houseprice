#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.cluster import FeatureAgglomeration
from sys import path
path.append ("../ingestion_program") # Contains libraries you will need
from data_manager import DataManager  # such as DataManager

from prepro import Preprocessor
input_dir = "../sample_data"
output_dir = "../resuts"

basename = 'Housing'
D = DataManager(basename, input_dir) # Load data
print("*** Original data ***")
print D

Prepro = Preprocessor()
    
X=np.copy(D.data['X_train'])
y=np.copy(D.data['Y_train'])
x_valid=np.copy(D.data['X_valid'])
x_test=np.copy(D.data['X_valid'])
    
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

# Preprocessing gives you opportunities of visualization:
# Scatter-plots of the 2 first principal components
# Scatter plots of pairs of features that are most relevant
import matplotlib.pyplot as plt
X = D.data['X_train']
Y = D.data['Y_train']
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()