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

model = datasets.get_model(PROCESSED_DATASETS['htru'])
c_tree = BATDepth(model, log = True)
c_tree.fit()