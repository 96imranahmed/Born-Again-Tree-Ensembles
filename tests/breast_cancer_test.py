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
#     'wpbc': datasets.get_data("../data/breast-cancer/wpbc.data", 'wpbc'),
#     'wdbc': datasets.get_data("../data/breast-cancer/wdbc.data", 'wdbc'),
#     'breast_cancer_yugoslavia': datasets.get_data("../data/breast-cancer/breast-cancer.data", 'breast_cancer_yugoslavia'),
    'breast_cancer_wisconsin': datasets.get_data("../data/breast-cancer/breast-cancer-wisconsin.data", 'breast_cancer_wisconsin'),
}

model = datasets.get_model(PROCESSED_DATASETS['breast_cancer_wisconsin'])
c_tree = BATDepth(model, log = False)
output_decision_tree = c_tree.fit()
test_predict = output_decision_tree.predict(test_data[test_data.columns.difference(['target'])].values)
print('BATDepth accuracy: {}'.format((test_predict == test_data['target']).sum()/len(test_predict)))