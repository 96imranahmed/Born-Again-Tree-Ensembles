import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout



class DecisionTree(object):
    d_tree_dict = None
    columns = None

    def __init__(self, dict_in, columns):
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

    def __get_node_label(self, details):
        if details[0] == 'leaf':
            return 'Return {}'.format(int(details[1]))
        else:
            c_col = str(details[1])
            if self.columns is not None: c_col = str(self.columns[int(details[1])])
            return '{} <= {}'.format(c_col, details[2])

    def visualise_tree(self, save_file):
        G = nx.DiGraph()
        label_dict = {}
        color_list = []
        if isinstance(self.d_tree_dict, list):
            G.add_node(tuple(self.d_tree_dict), **
                       {'label': self.__get_node_label(self.d_tree_dict)
                       , 'type':'leaf'})
            label_dict[tuple(self.d_tree_dict)] = self.__get_node_label(self.d_tree_dict)
            color_list.append('r')

        else:
            # Add root to graph
            uq_id = tuple(['Root'] + self.d_tree_dict['node'])
            G.add_node(uq_id, **{'label': self.__get_node_label(self.d_tree_dict['node']), 'type':'node'})
            label_dict[uq_id] = self.__get_node_label(self.d_tree_dict['node'])
            color_list.append('b')
            q = [
                (['Root'] + self.d_tree_dict['node'], self.d_tree_dict['left_child']),
                (['Root'] + self.d_tree_dict['node'], self.d_tree_dict['right_child'])
            ]
            while (len(q) > 0):
                c_entry=q.pop(0)
                if isinstance(c_entry[1], list):
                    uq_id = tuple(c_entry[0] + c_entry[1])
                    G.add_node(uq_id, **{'label': self.__get_node_label(c_entry[1]), 'type':'leaf'})
                    G.add_edge(tuple(c_entry[0]), uq_id)
                    label_dict[uq_id] = self.__get_node_label(c_entry[1])
                    color_list.append('r')
                else:
                    uq_id = tuple(c_entry[0] + c_entry[1]['node'])
                    G.add_node(uq_id, **{'label': self.__get_node_label(c_entry[1]['node']), 'type':'node'})
                    G.add_edge(tuple(c_entry[0]), uq_id)
                    label_dict[uq_id] = self.__get_node_label(c_entry[1]['node'])
                    color_list.append('b')
                    q.append(
                        (c_entry[0] + c_entry[1]['node'], c_entry[1]['left_child']))
                    q.append(
                        (c_entry[0] + c_entry[1]['node'], c_entry[1]['right_child']))

        # Print DiGraph to file
        pos = graphviz_layout(G, prog='dot')
        nx.draw(G, pos, 
        labels = label_dict, 
        node_color = color_list, 
        with_labels=True, 
        arrows=True,
        font_size = 8)
        plt.savefig(save_file)