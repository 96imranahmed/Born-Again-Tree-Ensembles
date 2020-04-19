import pandas as pd
import numpy as np

class DecisionTree(object):
    d_tree_dict = None

    def __init__(self, dict_in, columns):
        print(dict_in)
        self.d_tree_dict = dict_in
        self.columns = columns

    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy().tolist()
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        c_dict = self.d_tree_dict
        while True:
            if isinstance(c_dict, dict):
                split = c_dict['node']
                if xi[split[1]] <= split[-1]:
                    c_dict = c_dict['left_child']
                else:
                    c_dict = c_dict['right_child']
            else:
                return int(c_dict[1])