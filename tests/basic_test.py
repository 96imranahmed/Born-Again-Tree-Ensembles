import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import pandas as pd
import numpy as np

# Import files from source directory
import os
import sys
module_path = os.path.abspath(os.path.join('../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
import hyperplane, datasets
from model import BATDepth

data = pd.DataFrame( {
    'target' : [0,1,1,0,1,0,0,0,0,1,1,0],
    'x1' : [1, 1, 1, 3, 3, 3, 5.5, 5.5, 5.5, 8, 8, 8],
    'x2' : [1, 3, 10, 1, 3, 10, 1, 3, 5, 1, 3, 10]
})
model =  sklearn.ensemble.RandomForestClassifier(n_estimators = 10, 
                                                 max_depth=3,
                                                 max_features = len(data.columns) - 1).fit(data[data.columns.difference(['target'])], data['target'])

test_predict = model.predict_proba(data[data.columns.difference(['target'])])[:, 1]
fpr, tpr, _ = sklearn.metrics.roc_curve(data['target'], test_predict)
roc_auc = sklearn.metrics.auc(fpr, tpr)
print('ROC score: {}'.format(roc_auc))
print('Accuracy: {}'.format(sum((test_predict > 0.5) == data['target'])/len(test_predict)))

c_tree = BATDepth(model)
c_tree.fit()