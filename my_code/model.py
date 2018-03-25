'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import pickle
from prepro import preprocessor
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn import ensemble 

class model:
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        self.clf = ensemble.RandomForestRegressor()

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
        prepro=preprocessor()
        prepro.pipe(10)
        prepro.fit_transform(X,y)
        self.num_train_samples = len(X)
        if X.ndim>1: self.num_feat = len(X[0])
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = len(y)
        if y.ndim>1: self.num_labels = len(y[0])
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        self.clf.fit(X, y)
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
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        return self.clf.predict(X)

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self

if __name__ == "__main__":
    import imp
    from sklearn.model_selection import KFold
    from numpy import zeros, mean
    r2_score = imp.load_source('metric', "../scoring_program/my_metric.py").my_r2_score
    #defining the variables
    Xtrain = np.loadtxt("../../public_data/houseprice_train.data")
    ytrain = np.loadtxt("../../public_data/houseprice_train.solution")
    Xtest = np.loadtxt("../../public_data/houseprice_test.data")
    Xvalid = np.loadtxt("../../public_data/houseprice_valid.data")
    #defining the classifier
    classifier = model()
    classifier.fit(Xtrain, ytrain)
    y_hat_train = classifier.predict(Xtrain)
    training_error = r2_score(ytrain, y_hat_train)
    #cross validation
    n=3
    k=KFold(n_splits=n)
    k.get_n_splits(Xtrain)
    i=0
    scores = zeros(n)
    for l, m in k.split(Xtrain):
        Xtr, Xva = Xtrain[l], Xtrain[m]
        Ytr, Yva = ytrain[l], ytrain[m]
        mod = model()
        mod.fit(Xtr, Ytr)
        Yhat = mod.predict(Xva)
        scores[i] = r2_score(Yva, Yhat)
        print ('Fold', i+1, 'example metric = ', scores[i])
        i=i+1
    cross_validation_error = mean(scores)
    #results
    print("\nTraining scores: ", training_error)
    print ("Cross-Validation scores: ", cross_validation_error)
