import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

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
N_ESTIMATORS = 3

model =  sklearn.ensemble.RandomForestClassifier(n_estimators = N_ESTIMATORS, 
                                                 max_features = len(data.columns) - 1).fit(data[data.columns.difference(['target'])], data['target'])

print('Exporting {} trees of classifier...'.format(model.n_estimators))
fn=data.columns.difference(['target'])
fig, axes = plt.subplots(
    nrows = 1, 
    ncols = model.n_estimators, 
    figsize = (model.n_estimators * 5, 5), 
)
for index in range(0, model.n_estimators):
    tree.plot_tree(model.estimators_[index],
                   feature_names = fn, 
                   filled = True,
                   ax = axes[index])
    axes[index].set_title('Estimator: ' + str(index + 1))
plt.tight_layout()
fig.savefig('./outputs/basic_test_estimators.png', bbox_inches='tight')
plt.close(fig)

test_predict = model.predict_proba(data[data.columns.difference(['target'])])[:, 1]
fpr, tpr, _ = sklearn.metrics.roc_curve(data['target'], test_predict)
roc_auc = sklearn.metrics.auc(fpr, tpr)
print('ROC score: {}'.format(roc_auc))
print('Accuracy: {}'.format(sum((test_predict > 0.5) == data['target'])/len(test_predict)))

c_tree = BATDepth(model, log=False, columns = data.columns.difference(['target']).values)
output_decision_tree = c_tree.fit()
test_predict = output_decision_tree.predict(data[data.columns.difference(['target'])].values)
print('Final accuracy: {}'.format((test_predict == data['target']).sum()/len(test_predict)))

output_decision_tree.visualise_tree('./outputs/basic_test_batdepth.png')