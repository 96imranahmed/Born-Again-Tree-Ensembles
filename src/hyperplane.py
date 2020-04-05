import sklearn.tree
import sklearn.ensemble
import sklearn.exceptions


def __validate_input(model):
    if not (isinstance(model, sklearn.ensemble.RandomForestClassifier) or
            isinstance(model, sklearn.ensemble.RandomForestRegressor)):
        raise ValueError(
            'Expected model of type RandomForestClassifier or RandomForestRegressor, got: {}'.format(type(model)))
    if not hasattr(model, "estimators_"):
        raise sklearn.exceptions.NotFittedError('Model has not yet been fitted. Fit model first before calling estimator.')
    if model.n_outputs_ > 1: 
        raise NotImplementedError("Multiple outputs not supported")
    return True

def __extract_hyperplanes(model):   
    n_features = model.n_features_
    hyperplanes = [set()]*n_features

    for estimator in model.estimators_:
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        for i in range(estimator.tree_.node_count):
            if feature[i] == sklearn.tree._tree.TREE_UNDEFINED:
                continue
            hyperplanes[feature[i]].add(threshold[i])
    return [sorted(list(val)) for val in hyperplanes]
