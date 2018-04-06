'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.base import  BaseEstimator

from prepro import Preprocessor
from sklearn.pipeline import Pipeline


class model:
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        Model is the class called by Codalab.
        This class must have at least a method "fit" and a method "predict".
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        # The model should be defined in the constructor
        self.mod = Pipeline([
                 ('preprocessing', Preprocessor()),
                ('predictor', Predictor())
                ])
        print("MODEL=" + self.mod.__str__())

    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''
        
        # Do not remove the "debug code" this is for "defensive programming"
        self.num_train_samples = len(X)
        if X.ndim>1: self.num_feat = len(X[0])
        print("FIT: dim(X)= [{}, {}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = len(y)
        if y.ndim>1: self.num_labels = len(y[0])
        print("FIT: dim(y)= [{}, {}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        
        # Thi sis where training happens
        self.mod.fit(X, y)
        self.is_trained=True

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        num_test_samples = len(X)
        if X.ndim>1: num_feat = len(X[0])
        print("PREDICT input: dim(X)= [{}, {}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
  
        Y = self.mod.predict(X)
        
        num_labels = 1
        if Y.ndim>1: num_labels = len(Y[0])
        print("PREDICT output: dim(y)= [{}, {}]".format(num_test_samples, self.num_labels))
        if (self.num_labels != num_labels):
            print("ARRGH: number of labels in X does not match training data!")
            
        return Y

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
        
class Predictor(BaseEstimator):
    '''Predictor: modify this class to create a predictor of
    your choice. This could be your own algorithm, of one for the scikit-learn
    models, for which you choose the hyper-parameters.'''
    def __init__(self):
        '''This method initializes the predictor.'''
        self.mod = BaggingRegressor(base_estimator=RandomForestRegressor(n_estimators=50))
        print("PREDICTOR=" + self.mod.__str__())

    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        self.mod = self.mod.fit(X, y)
        return self

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.mod.predict(X)

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
 
from sys import argv, path      
if __name__=="__main__":
    # Modify this class to serve as test
        
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../../public_data" # A remplacer par le bon chemin
        output_dir = "../results" # A remplacer par le bon chemin
        code_dir = "../ingestion" # A remplacer par le bon chemin
        metric_dir = "../scoring_program" # A remplacer par le bon chemin
    else:
        input_dir = argv[1]
        output_dir = argv[2]
        code_dir = argv[3]
        metric_dir = argv[4]
        
    path.append (code_dir)
    path.append (metric_dir)
    
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    from numpy import zeros
    import matplotlib.pyplot as plt
    
    with open(metric_dir + '/metric.txt', 'r') as f:
        metric_name = f.readline().strip()
        import libscores, my_metric
        try:
            scoring_function = getattr(libscores, metric_name)
        except:
            scoring_function = getattr(my_metric, metric_name)
    print 'Using scoring metric:', metric_name
            
    from data_manager import DataManager
    basename = 'houseprice'
    D = DataManager(basename, input_dir) # Load data
    print D
    
    # Here we define models and compare them
    model_dict = {
            'RandomForest':RandomForestRegressor(n_estimators=50),
            'RandomForest+BaggingRegressor': BaggingRegressor(base_estimator=RandomForestRegressor(n_estimators=50))
            }
    k=0
    training_score = zeros([len(model_dict)])
    cv_score = zeros([len(model_dict)])
    for key in model_dict:
        mymodel = model_dict[key]
        print("\n\n *** Model {:s}:{:s}".format(key,model_dict[key].__str__()))
 
        # Train
        print("Training")
        X_train = D.data['X_train']
        Y_train = D.data['Y_train']
        mymodel.fit(X_train, Y_train)
    
        # Predictions on training data
        print("Predicting")
        Ypred_tr = mymodel.predict(X_train)
        
        # Cross-validation predictions
        print("Cross-validating")
        n = 10 # 10-fold cross-validation
        kf = KFold(n_splits=n)
        kf.get_n_splits(X_train)
        Ypred_cv = zeros(Ypred_tr.shape)
        i=1
        for train_index, test_index in kf.split(X_train):
            print("Fold {}".format(i))
            Xtr, Xva = X_train[train_index], X_train[test_index]
            Ytr, Yva = Y_train[train_index], Y_train[test_index]
            mymodel.fit(Xtr, Ytr)
            Ypred_cv[test_index] = mymodel.predict(Xva)
            i = i+1
            

        # Compute and print performance
        training_score[k] = scoring_function(Y_train, Ypred_tr)
        cv_score[k] = scoring_function(Y_train, Ypred_cv)
        
        print("\nRESULTS FOR SCORE {:s}".format(metric_name))
        print("TRAINING SCORE= {:f}".format(training_score[k]))
        print("CV SCORE= {:f}".format(cv_score[k]))
        
        k = k+1
    
    print("\nGlobal results : \n\n")
    
    print("TRAINING SCORE WITH THE DIFFERENT REGRESSORS: {}".format(dict(zip(model_dict.keys(),training_score))))
    print("X-VALIDATION WITH THE DIFFERENT REGRESSORS: {}".format(dict(zip(model_dict.keys(),cv_score))))

    print("RESULTS : {}".format(training_score))

    
    #plt.figure()
    #plt.plot(training_score)