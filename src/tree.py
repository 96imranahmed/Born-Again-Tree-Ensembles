import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import collections


class DecisionTree(object):
    d_tree_dict = None
    columns = None

    def __init__(self, dict_in, columns):
        self.d_tree_dict = self.__get_node_ids(dict_in, 'root')
        self.columns = columns

    def predict(self, x, return_path = False):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy().tolist()
        output_array = []
        path_counter = collections.Counter()
        for xi in x:
            if return_path:
                c_pred, c_path = self.predict_row(xi, True)
                output_array.append(c_pred)
                path_counter.update(c_path)
            else:
                output_array.append(self.predict_row(xi, False))
        if return_path:
            return np.array(output_array), dict(path_counter)
        else:
            return np.array(output_array)

    def predict_row(self, xi, return_path = False):
        c_dict = self.d_tree_dict
        c_path = set()
        while True:
            if isinstance(c_dict, dict):
                split = c_dict['node']
                c_path.add(split[-1])
                if xi[split[1]] <= split[2]:
                    c_dict = c_dict['left_child']
                else:
                    c_dict = c_dict['right_child']
            else:
                c_path.add(c_dict[-1])
                if return_path:
                    return int(c_dict[1]), c_path
                else:
                    return int(c_dict[1])

    def __get_node_label(self, details):
        if details[0] == 'leaf':
            return 'Return {}'.format(int(details[1]))
        else:
            c_col = str(details[1])
            if self.columns is not None: c_col = str(self.columns[int(details[1])])
            return '{} <= {}'.format(c_col, details[2])
        
    def __get_node_ids(self, dict_in, stub):
        if isinstance(dict_in, list):
            return dict_in + [stub + '|' + str(dict_in)]
        else:
            updated_stub = str(stub) + '|' + str(dict_in['node'])
            return {
                'node': dict_in['node'] + [updated_stub],
                'left_child': self.__get_node_ids(dict_in['left_child'], updated_stub),
                'right_child': self.__get_node_ids(dict_in['right_child'], updated_stub)
            }

    def visualise_tree(self, save_file):
        G = nx.DiGraph()
        label_dict = {}
        color_list = []
        if isinstance(self.d_tree_dict, list):
            G.add_node(self.d_tree_dict[-1])
            label_dict[self.d_tree_dict[-1]] = self.__get_node_label(self.d_tree_dict)
            color_list.append('r')

        else:
            # Add root to graph
            uq_id = self.d_tree_dict['node'][-1]
            G.add_node(uq_id)
            label_dict[uq_id] = self.__get_node_label(self.d_tree_dict['node'])
            color_list.append('b') # Add a Node
            q = [
                (self.d_tree_dict['node'][-1], self.d_tree_dict['left_child']),
                (self.d_tree_dict['node'][-1], self.d_tree_dict['right_child'])
            ]
            while (len(q) > 0):
                parent_node_id, c_node  =q.pop(0)
                if isinstance(c_node, list):
                    G.add_node(c_node[-1])
                    G.add_edge(parent_node_id, c_node[-1])
                    label_dict[c_node[-1]] = self.__get_node_label(c_node)
                    color_list.append('r') # Add a leaf
                else:
                    G.add_node(c_node['node'][-1])
                    G.add_edge(parent_node_id, c_node['node'][-1])
                    label_dict[c_node['node'][-1]] = self.__get_node_label(c_node['node'])
                    color_list.append('b') # Add a node
                    q.append(
                        (c_node['node'][-1], c_node['left_child']))
                    q.append(
                        (c_node['node'][-1], c_node['right_child']))

        # Print DiGraph to file
        pos = graphviz_layout(G, prog='dot')
        nx.draw(G, pos, 
        labels = label_dict, 
        node_color = color_list, 
        with_labels=True, 
        arrows=True,
        font_size = 8)
        plt.savefig(save_file)
    
    def prune(self, train_data):
        _, path = self.predict(train_data, True)
        raise NotImplementedError('Not yet implemented (higher priority to make the code more efficient)')