import pickle
import random
from pprint import pprint

import pandas as pd
import torch
from sklearn.utils import class_weight
import numpy as np

# from feature_utils import inchisToMols, mol2graph, show_atoms_features
from random import shuffle
from prettytable import PrettyTable

from utils.feature_utils import inchisToMols, mol2graph, show_atoms_features


# carga el dataset sin separar en train, test, validataion
def load_common_tags_dataset_no_split(dataset_path=None):
    # pl.seed_everything(42)
    train = pd.read_csv(dataset_path).sample(frac=1)

    train_labels = [0 for _ in range(train.shape[0])]
    train_mols = molecules_dataset_to_graph_list(train, train_labels)
    train_mols = remove_invalid_mols(train_mols, train_labels)

    return train_mols


def molecules_dataset_to_graph_list(dataset, labels):
    inchis = dataset['inchi']
    names = dataset['common_name']
    mol_inchis = inchisToMols(inchis)
    # mols = [mol2graph(m) for m in mols]
    mols = []
    for mol_label_name in zip(mol_inchis, labels, names):
        mols.append(mol2graph(mol_label_name[0], mol_label_name[1], mol_label_name[2]))

    return mols


def molecules_dataset_to_graph_list_nx(dataset, labels):
    mols = []
    for mol_label in zip(dataset, labels):
        m = mol2graph_NX(mol_label[0], mol_label[1])
        if m and m.num_edges > 0:
            n = m.num_nodes
            if m.num_edges > (n * (n - 1)) / 2:
                print('error', m.num_edges, m.num_nodes)
            mols.append(m)

    return mols


def load_sweet_bitter_dataset(sample=1):
    # pl.seed_everything(42)
    sweet = pd.read_csv('../dataset/sulfur_floral/sweet_and_something_else.csv')
    bitter = pd.read_csv('../dataset/sulfur_floral/bitter.csv').head(635)  # sample(n=635)

    sweet['sweet'] = 1
    bitter['sweet'] = 0

    train_test_dataset = pd.concat([sweet, bitter]).head(66335)  # sample(frac=1)

    y = train_test_dataset.sweet.to_numpy()
    labels = [torch.tensor([label], dtype=torch.long) for label in y]

    mols = molecules_dataset_to_graph_list(train_test_dataset, labels)
    mols = remove_invalid_mols(mols, y)

    show_features(train_test_dataset)

    train, validation, test = train_validation_test(mols, test_ratio=0.4)

    show_dataset_metrics(test, train, validation)

    return train, validation, test


def show_features(train_test_dataset, key='inchi'):
    inchis = train_test_dataset[key]
    mol_inchis = inchisToMols(inchis)
    show_atoms_features(mol_inchis)


def load_odor_dataset(sample=1, augment_train_positives_by_factor=0, augment_train_negatives_by_factor=0):
    # pl.seed_everything(42)
    odorless = pd.read_csv('../dataset/odor/odorless.csv')
    not_odorless = pd.read_csv('../dataset/odor/not_odorless.csv').head(635)  # sample(n=635)

    odorless['odorless'] = 1
    not_odorless['odorless'] = 0

    train_test_dataset = pd.concat([odorless, not_odorless]).head(66335)  # sample(frac=1)
    show_features(train_test_dataset)

    y = train_test_dataset.odorless.to_numpy()
    labels = [torch.tensor([label], dtype=torch.long) for label in y]

    mols = molecules_dataset_to_graph_list(train_test_dataset, labels)
    mols = remove_invalid_mols(mols, y)

    train, validation, test = train_validation_test(mols, test_ratio=0.5,
                                                    augment_train_positives_by_factor=augment_train_positives_by_factor,
                                                    augment_train_negatives_by_factor=augment_train_negatives_by_factor)

    show_dataset_metrics(test, train, validation)

    return train, validation, test


def split_list(a_list):
    half = len(a_list) // 2
    return a_list[:half], a_list[half:]


def load_dataset(sample=1, positive_key='has_sweet', augment_positives_ratio=0,
                 dataset_path='../dataset/molecules_df.csv'):
    # pl.seed_everything(42)
    dataset = pd.read_csv(dataset_path).sample(frac=sample)

    y = dataset[positive_key].to_numpy()
    labels = [torch.tensor([label], dtype=torch.long) for label in y]

    mols = molecules_dataset_to_graph_list(dataset, labels)
    mols = remove_invalid_mols(mols, y)

    train, validation, test = train_validation_test(mols, test_ratio=0.3,
                                                    augment_train_positives_by_factor=augment_positives_ratio)
    show_dataset_metrics(test, train, validation)

    return train, validation, test


def load_common_tags_dataset_by_tags(sample=1, augment_positives_ratio=0,
                                     dataset_path='../dataset/common_tags/common_tags.csv',
                                     tags=['fruity', 'green', 'bitter']):
    # pl.seed_everything(42)
    train_datasets = []
    test_datasets = []
    for tag in tags:
        train = pd.read_csv(f'../dataset/common_tags/train/{tag}_train.csv')
        test = pd.read_csv(f'../dataset/common_tags/test/{tag}_test.csv')

        train_datasets.append(train)
        test_datasets.append(test)

    train_d = pd.concat(train_datasets)
    test_d = pd.concat(test_datasets)
    print('traind', len(train_d))

    train = train_d.drop_duplicates(subset=['pubchem_id'])
    test = test_d.drop_duplicates(subset=['pubchem_id'])
    print('train', len(train))

    train_labels = get_multilabels(train, tags)
    test_labels = get_multilabels(test, tags)

    train_mols = molecules_dataset_to_graph_list(train, train_labels)
    train_mols = remove_invalid_mols(train_mols, train_labels)

    test_mols = molecules_dataset_to_graph_list(test, test_labels)
    test_mols = remove_invalid_mols(test_mols, test_labels)

    validation_mols, test_mols = split_list(test_mols)

    return train_mols, validation_mols, test_mols


def load_common_tags_dataset(dataset_path='../dataset/common_tags/unbalanced',
                             tags=['fruity', 'green', 'woody', 'floral', 'bitter', 'herbal'],
                             test_set_path=None,
                             bypas_intersections=False,
                             show_tags=True):
    # pl.seed_everything(42)
    train = pd.read_csv(f'{dataset_path}/unbalanced_common_tags_train.csv').sample(frac=1)
    if test_set_path:
        test = pd.read_csv(test_set_path).sample(frac=1)
    else:
        test = pd.read_csv(f'{dataset_path}/unbalanced_common_tags_test.csv').sample(frac=1)
    validation = pd.read_csv(f'{dataset_path}/unbalanced_common_tags_validation.csv').sample(frac=1)

    if not bypas_intersections:
        check_datasets_intersection(test, train, validation)

    tag_weights = {}
    for tag in tags:
        tag_labels = train[tag]
        tag_weights_ = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(tag_labels),
                                                         y=tag_labels.to_numpy())
        tag_weights[tag] = tag_weights_

    print('TAG WEIGHTS:', '\n')
    pprint(tag_weights)

    train_labels = get_multilabels(train, tags)
    test_labels = get_multilabels(test, tags)
    validation_labels = get_multilabels(validation, tags)

    train_mols = molecules_dataset_to_graph_list(train, train_labels)
    train_mols = remove_invalid_mols(train_mols, train_labels)

    test_mols = molecules_dataset_to_graph_list(test, test_labels)
    test_mols = remove_invalid_mols(test_mols, test_labels)

    validation_mols = molecules_dataset_to_graph_list(validation, validation_labels)
    validation_mols = remove_invalid_mols(validation_mols, validation_labels)

    if show_tags:
        show_tags_neg_pos(tags, test, test_mols, train, train_mols, validation, validation_mols)

    return train_mols, validation_mols, test_mols, tag_weights


def load_common_tags_dataset_nx(sample=1, augment_positives_ratio=0,
                                dataset_path='../dataset/common_tags/balanced_tags',
                                tags=['fruity', 'green', 'woody', 'floral', 'bitter', 'herbal']):
    # pl.seed_everything(42)
    with open(f'{dataset_path}/common_tags_train_nx.p', 'rb') as f:
        train_nx = pickle.load(f)
        train_ids = {'pubchem_id': [id_mol[0] for id_mol in train_nx]}
        train_ids = pd.DataFrame(train_ids)
    with open(f'{dataset_path}/common_tags_test_nx.p', 'rb') as f:
        test_nx = pickle.load(f)
        test_ids = {'pubchem_id': [id_mol[0] for id_mol in test_nx]}
        test_ids = pd.DataFrame(test_ids)
    with open(f'{dataset_path}/common_tags_validation_nx.p', 'rb') as f:
        validation_nx = pickle.load(f)
        validation_ids = {'pubchem_id': [id_mol[0] for id_mol in validation_nx]}
        validation_ids = pd.DataFrame(validation_ids)

    train = pd.read_csv(f'{dataset_path}/common_tags_train.csv')
    train = train_ids.merge(train, how='inner', on='pubchem_id')

    test = pd.read_csv(f'{dataset_path}/common_tags_test.csv')
    test = test_ids.merge(test, how='inner', on='pubchem_id')

    validation = pd.read_csv(f'{dataset_path}/common_tags_validation.csv')
    validation = validation_ids.merge(validation, how='inner', on='pubchem_id')

    check_datasets_intersection(test, train, validation)

    tag_weights = {}
    for tag in tags:
        tag_labels = train[tag]
        tag_weights_ = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(tag_labels),
                                                         y=tag_labels.to_numpy())
        print('TAG WEIGHTS: ' + tag)
        pprint(tag_weights)
        tag_weights[tag] = tag_weights_

    train_labels = get_multilabels(train, tags)
    test_labels = get_multilabels(test, tags)
    validation_labels = get_multilabels(validation, tags)

    train_nx = [t[1] for t in train_nx]
    train_mols = molecules_dataset_to_graph_list_nx(train_nx, train_labels)
    train_mols = remove_invalid_mols_nx(train_mols, train_labels)

    test_nx = [t[1] for t in test_nx]
    test_mols = molecules_dataset_to_graph_list_nx(test_nx, test_labels)
    test_mols = remove_invalid_mols_nx(test_mols, test_labels)

    validation_nx = sorted(validation_nx, key=lambda x: x[1].number_of_edges(), reverse=True)
    validation_nx = [t[1] for t in validation_nx]
    validation_mols = molecules_dataset_to_graph_list_nx(validation_nx, validation_labels)
    validation_mols = remove_invalid_mols_nx(validation_mols, validation_labels)

    show_tags_neg_pos(tags, test, test_mols, train, train_mols, validation, validation_mols)

    return train_mols, validation_mols, test_mols, tag_weights


def show_tags_neg_pos(tags, test, test_mols, train, train_mols, validation, validation_mols):
    table = PrettyTable(["Tag", "Pos", "Neg"])
    table.title = 'Train Counts'
    for tag in tags:
        table.add_row([tag, train[tag].value_counts()[1], train[tag].value_counts()[0]])
    print(table)
    table = PrettyTable(["Tag", "Pos", "Neg"])
    table.title = 'Test Counts'
    for tag in tags:
        # pos = test[tag].value_counts()[1]
        # neg = test[tag].value_counts()[0]
        table.add_row([tag, test[tag].value_counts()[1], test[tag].value_counts()[0]])
    print(table)
    table = PrettyTable(["Tag", "Pos", "Neg"])
    table.title = 'Validation Counts'
    for tag in tags:
        table.add_row([tag, validation[tag].value_counts()[1], validation[tag].value_counts()[0]])
    print(table)
    print('TRAIN LEN', len(train_mols))
    print('VALIDATION LEN', len(validation_mols))
    print('TEST LEN', len(test_mols))


def check_datasets_intersection(test, train, validation):
    print('Checking intersections')
    test_train = pd.merge(train, test, how='inner', on=['pubchem_id'])
    test_validation = pd.merge(validation, test, how='inner', on=['pubchem_id'])
    train_validation = pd.merge(train, validation, how='inner', on=['pubchem_id'])
    if len(test_train) > 0:
        print(test_train)
        raise Exception('Train and test datasets have intersections')
    if len(test_validation) > 0:
        print(test_validation)
        raise Exception('Validation and test datasets have intersections')
    if len(train_validation) > 0:
        print(train_validation)
        raise Exception('Train and validatoin datasets have intersections')
    print('All good no intersections found')


def get_multilabels(dataset, tags):
    tags_labels = {}
    for tag in tags:
        has_tag = dataset[tag].to_numpy()
        has_tag = [torch.tensor([label], dtype=torch.long) for label in has_tag]

        tags_labels[tag] = has_tag
    labels = []
    for i in range((len(dataset))):
        label = {}
        for tag in tags:
            label[tag] = tags_labels[tag][i]
        # label = {
        #    'fruity': has_fruity[i],
        #    'herbal': has_green[i]
        # }

        labels.append(label)
    return labels


def remove_invalid_mols(mols, y):
    mols_labeled = zip(mols, y)
    mols_labeled_not_nulls = []
    for labeled_mol in mols_labeled:
        labeled_mol_mol = labeled_mol[0]
        if labeled_mol_mol is not None:
            mols_labeled_not_nulls.append(labeled_mol)
    mols = [x[0] for x in mols_labeled_not_nulls]

    return mols


def remove_invalid_mols_nx(mols, y):
    mols_labeled = zip(mols, y)
    mols_labeled_not_nulls = []
    for labeled_mol in mols_labeled:
        labeled_mol_mol = labeled_mol[0]
        if labeled_mol_mol is not None and labeled_mol_mol.num_edges > 1:
            mols_labeled_not_nulls.append(labeled_mol)

    mols = [x[0] for x in mols_labeled_not_nulls]

    return mols


def show_dataset_metrics(test, train, validation):
    train_negs = [x for x in train if x.y == 0]
    train_pos = [x for x in train if x.y == 1]

    validation_negs = [x for x in validation if x.y == 0]
    validation_pos = [x for x in validation if x.y == 1]

    test_negs = [x for x in test if x.y == 0]
    test_pos = [x for x in test if x.y == 1]

    total_records = len(train) + len(test) + len(validation)

    train_neg_ratio = round(len(train_negs) * 100 / total_records, 2)
    train_pos_ratio = round(len(train_pos) * 100 / total_records, 2)

    validation_neg_ratio = round(len(validation_negs) * 100 / total_records, 2)
    validation_pos_ratio = round(len(validation_pos) * 100 / total_records, 2)

    test_neg_ratio = round(len(test_negs) * 100 / total_records, 2)
    test_pos_ratio = round(len(test_pos) * 100 / total_records, 2)

    print(
        'Neg/Pos ratio: ' + f'{str(round(train_neg_ratio + validation_neg_ratio + test_neg_ratio, 2))}/{str(round(train_pos_ratio + validation_pos_ratio + test_pos_ratio, 2))}\n')

    print('- Train: ' + str(len(train)) + f' --> {round(len(train) * 100 / total_records, 2)}%')
    print('  * Negs: ' + str(len(train_negs)) + f' --> {train_neg_ratio}%')
    print('  * Pos: ' + str(len(train_pos)) + f' --> {train_pos_ratio}%' + '\n')

    print('- Validation: ' + str(len(validation)) + f' --> {round(len(validation) * 100 / total_records, 2)}%')
    print('  * Negs: ' + str(len(validation_negs)) + f' --> {validation_neg_ratio}%')
    print('  * Pos: ' + str(len(validation_pos)) + f' --> {validation_pos_ratio}%' + '\n')
    print('- Test: ' + str(len(test)) + f' --> {round(len(test) * 100 / total_records, 2)}%')
    print('  * Negs: ' + str(len(test_negs)) + f' --> {test_neg_ratio}%')
    print('  * Pos: ' + str(len(test_pos)) + f' --> {test_pos_ratio}%')


# retorna train, validation, test donde la proporcion de validation y test equivale al test ratio/2 en cada uno
def train_validation_test(mols, test_ratio=0.3, augment_train_positives_by_factor=0,
                          augment_train_negatives_by_factor=0):
    pos = []
    neg = []

    for mol in mols:
        if mol.y == 1:
            pos.append(mol)
        else:
            neg.append(mol)

    train_ratio = 1 - test_ratio

    # shuffle(pos)
    # shuffle(neg)

    train_pos = pos[:int(len(pos) * train_ratio)]
    test_pos = pos[-int(len(pos) * test_ratio):]

    train_neg = neg[:int(len(neg) * train_ratio)]
    test_neg = neg[-int(len(neg) * test_ratio):]

    for x in range(augment_train_positives_by_factor):
        train_pos += train_pos

    for x in range(augment_train_negatives_by_factor):
        train_neg += train_neg

    train = train_pos + train_neg

    validation_pos, test_pos = split_list(test_pos)
    validation_neg, test_neg = split_list(test_neg)

    test = test_pos + test_neg
    validation = validation_pos + validation_neg

    shuffle(train)
    shuffle(validation)
    shuffle(test)

    return train, validation, test


def train_validation_test_multilabel(mols, test_ratio=0.3, augment_train_positives_by_factor=0,
                                     augment_train_negatives_by_factor=0):
    train_ratio = 1 - test_ratio

    # shuffle(pos)
    # shuffle(neg)

    train = mols[:int(len(mols) * train_ratio)]
    test = mols[-int(len(mols) * test_ratio):]

    validation, test = split_list(test)

    shuffle(train)
    shuffle(validation)
    shuffle(test)

    return train, validation, test


def load_sulfur_floral_dataset(class_0='bitter', class_1='sulfur', sample_class_0=None, sample_class_1=None):
    # pl.seed_everything(42)
    floral = pd.read_csv(f'../dataset/sulfur_floral/{class_0}.csv')
    sulfur = pd.read_csv(f'../dataset/sulfur_floral/{class_1}.csv')

    if sample_class_0 is not None:
        floral = floral.sample(n=sample_class_0)

    if sample_class_1 is not None:
        sulfur = sulfur.sample(n=sample_class_1)

    train_test_dataset = pd.concat([floral, sulfur])

    y = train_test_dataset[class_1].to_numpy()
    labels = [torch.tensor([label], dtype=torch.long) for label in y]

    mols = molecules_dataset_to_graph_list(train_test_dataset, labels)
    mols = remove_invalid_mols(mols, y)

    show_features(train_test_dataset)

    train, validation, test = train_validation_test(mols, test_ratio=0.4)

    # show_dataset_metrics(test, train, validation)

    return train, validation, test


def load_sulfur_floral_dataset_nx(class_0='sweet_and_something_else', class_1='bitter', sample_class_0=None,
                                  sample_class_1=None):
    # pl.seed_everything(42)

    with open(f'../dataset/sulfur_floral/{class_0}.p', 'rb') as f:
        class_0_nx = pickle.load(f)
        class_0_nx_dict = {'pubchem_id': [id_mol[0] for id_mol in class_0_nx],
                           'graph': [id_mol[1] for id_mol in class_0_nx],
                           'name': [id_mol[2] for id_mol in class_0_nx],
                           'class': [0 for x in range(len(class_0_nx))]}

        class_0_nx = pd.DataFrame(class_0_nx_dict)

        if sample_class_0 is not None:
            class_0_nx = class_0_nx.head(sample_class_0)

    with open(f'../dataset/sulfur_floral/{class_1}.p', 'rb') as f:
        class_1_nx = pickle.load(f)
        class_1_nx_dict = {'pubchem_id': [id_mol[0] for id_mol in class_1_nx],
                           'graph': [id_mol[1] for id_mol in class_1_nx],
                           'name': [id_mol[2] for id_mol in class_1_nx],
                           'class': [1 for x in range(len(class_1_nx))]}

        class_1_nx = pd.DataFrame(class_1_nx_dict)

        if sample_class_1 is not None:
            class_1_nx = class_1_nx.head(sample_class_1)

    dataset_0_mols = list(class_0_nx['graph'])
    y_0 = class_0_nx['class'].to_numpy()
    labels_0 = [torch.tensor([label], dtype=torch.long) for label in y_0]
    mols_0 = molecules_dataset_to_graph_list_nx(dataset_0_mols, labels_0)

    dataset_1_mols = list(class_1_nx['graph'])
    y_1 = class_1_nx['class'].to_numpy()
    labels_1 = [torch.tensor([label], dtype=torch.long) for label in y_1]
    mols_1 = molecules_dataset_to_graph_list_nx(dataset_1_mols, labels_1)
    # dataset = pd.concat([class_0_nx, class_1_nx])
    # dataset_mols = list(dataset['graph'])

    # y = dataset['class'].to_numpy()
    # labels = [torch.tensor([label], dtype=torch.long) for label in y]

    # mols = molecules_dataset_to_graph_list_nx(dataset_mols, labels)

    # ACA YA SON OBJETOS TIPO "DATA"
    # mols = remove_invalid_mols_nx(mols, y)

    len1 = len(mols_0)
    len2 = len(mols_1)
    l_min = min(len1, len2)
    mols = mols_0[:l_min] + mols_1[: l_min]
    random.shuffle(mols)

    train, validation, test = train_validation_test(mols, test_ratio=0.4)

    # show_dataset_metrics(test, train, validation)

    return train, validation, test
