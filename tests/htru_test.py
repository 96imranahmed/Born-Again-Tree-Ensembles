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

PROCESSED_DATASETS = {
    'htru': datasets.get_data("../data/htru/HTRU_2.csv", 'htru'),
}

model, train_data, test_data = datasets.get_model(PROCESSED_DATASETS['htru'])
c_tree = BATDepth(model, log = True)
output_decision_tree = c_tree.fit()
test_predict = output_decision_tree.predict(test_data[test_data.columns.difference(['target'])].values)
print('Final accuracy: {}'.format((test_predict == test_data['target']).sum()/len(test_predict)))