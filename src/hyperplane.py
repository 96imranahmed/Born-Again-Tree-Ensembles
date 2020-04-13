import sklearn.tree
import sklearn.ensemble
import sklearn.exceptions
import numpy as np

def __validate_input(model):
    if not (isinstance(model, sklearn.ensemble.RandomForestClassifier)):
            raise ValueError('Expected model of type RandomForestClassifier, got: {}'.format(type(model)))
    if not hasattr(model, "estimators_"):
        raise sklearn.exceptions.NotFittedError('Model has not yet been fitted. Fit model first before calling estimator.')
    if model.n_outputs_ > 1: 
        raise NotImplementedError("Multiple outputs not supported")
    return True

def extract_hyperplanes(model):   
    n_features = model.n_features_
    hyperplanes = [set() for i in range(n_features)]

    for estimator in model.estimators_:
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        for i in range(estimator.tree_.node_count):
            if feature[i] != sklearn.tree._tree.TREE_UNDEFINED:
                hyperplanes[feature[i]].add(threshold[i])
    hyperplanes = [sorted(list(val)) for val in hyperplanes]
    print('Hyperplane z-bounds: ', get_bounds(hyperplanes))
    return hyperplanes

def get_bounds(hyperplanes):
    return np.ones([len(hyperplanes)], dtype = int), np.array([len(i) + 1 for i in hyperplanes], dtype = int)