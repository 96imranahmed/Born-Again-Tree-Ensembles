import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_htru_dataset(data_loc):
    data = pd.read_csv(data_loc, 
                       header = None)
    columns = [
        'mean_int_profile', 
        'std_int_profile', 
        'excess_kurtosis_int_profile', 
        'skewness_int_profile',
        'mean_dm_snr',
        'std_dm_snr',
        'excess_kurtosis_dm_snr', 
        'skewness_dm_snr',
        'target'
    ]
    data.columns = columns
    return data

def load_wpbc_dataset(data_loc):
    data = pd.read_csv(data_loc,
                       header=None).replace("?", np.nan)
    columns_stack = [
        'radius',
        'texture',
        'perimeter',
        'area',
        'smoothness',
        'compactness',
        'concavity',
        'concave_points',
        'symmetry',
        'fractal_dimension'
    ]

    columns_index = [
        'ID', 'target', 'time'
    ]

    columns_joint = [
        'tumor_size', 'lymph_node_status'
    ]

    data.columns = columns_index + ['mean_'+i for i in columns_stack] + [
        'std_' + i for i in columns_stack] + ['max_'+i for i in columns_stack] + columns_joint
    data = data.set_index('ID')
    data['target'] = data['target'].map({'R': 1, 'N': 0})
    return data[data.columns.difference(['time'])].dropna()


def load_wdbc_dataset(data_loc):
    data = pd.read_csv(data_loc,
                       header=None).replace("?", np.nan)
    columns_stack = [
        'radius',
        'texture',
        'perimeter',
        'area',
        'smoothness',
        'compactness',
        'concavity',
        'concave_points',
        'symmetry',
        'fractal_dimension'
    ]

    columns_index = [
        'ID', 'target'
    ]

    data.columns = columns_index + ['mean_'+i for i in columns_stack] + [
        'std_' + i for i in columns_stack] + ['max_'+i for i in columns_stack]
    data = data.set_index('ID')
    data['target'] = data['target'].map({'M': 1, 'B': 0})
    return data.dropna()


def load_wisconsin_bc_dataset(data_loc):
    data = pd.read_csv(data_loc,
                       header=0,
                       names=['ID',
                              'clump_thickness',
                              'cell_size',
                              'cell_shape',
                              'marginal_adhesion',
                              'epithelial_cell_size',
                              'bare_nuclei',
                              'bland_chromatin',
                              'normal_nucleoli',
                              'mitoses',
                              'target']).replace("?", np.nan).dropna()
    data = data[data.columns.difference(['ID', 'time'])]
    data['target'] = data['target'].map({2: 0, 4: 1})
    return data.dropna()


def load_yugoslavia_bc_dataset(data_loc):
    data = pd.read_csv(data_loc,
                       header=0,
                       names=[
                           'Class',
                           'age',
                           'menopause',
                           'tumor-size',
                           'inv-nodes',
                           'node-caps',
                           'deg-malig',
                           'breast',
                           'breast-quad',
                           'irradiat'
                       ]).replace("?", np.nan).dropna()
    data = pd.get_dummies(
        data,
        columns=['menopause',
                 'node-caps',
                 'breast',
                 'breast-quad',
                 'irradiat'],
        drop_first=True
    ).rename(columns={'Class': 'target'})
    data['target'] = data['target'].map(
        {'no-recurrence-events': 0, 'recurrence-events': 1})
    data['age'] = data['age'].apply(lambda x: int(x.split('-')[0]))
    data['tumor-size'] = data['tumor-size'].apply(
        lambda x: int(x.split('-')[0]))
    data['inv-nodes'] = data['inv-nodes'].apply(lambda x: int(x.split('-')[0]))
    return data.dropna()


MAPPING_DICT = {
    'wpbc': load_wpbc_dataset,
    'wdbc': load_wdbc_dataset,
    'breast_cancer_wisconsin': load_wisconsin_bc_dataset,
    'breast_cancer_yugoslavia': load_yugoslavia_bc_dataset,
    'htru': load_htru_dataset
}

def get_data(data_loc,
             dataset_name):
    if dataset_name not in set(MAPPING_DICT.keys()):
        raise ValueError("Dataset '{}' not recognised - select from '{}'".format(
            dataset_name,
            MAPPING_DICT.keys())
        )
    data = MAPPING_DICT[dataset_name](data_loc)
    return data

def get_model(data, print_plot = False):
    train_data, test_data = sklearn.model_selection.train_test_split(data, test_size = 0.2)
    model =  sklearn.ensemble.RandomForestClassifier(n_estimators = 2, 
                                                     max_depth=3,
                                                     max_features = len(data.columns) - 1).fit(train_data[train_data.columns.difference(['target'])], train_data['target'])

    test_predict = model.predict_proba(test_data[test_data.columns.difference(['target'])])[:, 1]
    fpr, tpr, _ = sklearn.metrics.roc_curve(test_data['target'], test_predict)
    roc_auc = sklearn.metrics.auc(fpr, tpr)

    if print_plot:
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                lw=2, label='ROC curve (area = {})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    print('RF ROC score: {}'.format(roc_auc))
    print('RF Accuracy: {}'.format(sum((test_predict > 0.5) == test_data['target'])/len(test_predict)))
    return model, train_data, test_data