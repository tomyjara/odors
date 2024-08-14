import datetime
from copy import deepcopy

import torch
from matplotlib import pyplot
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix
from torch_geometric.data import DataLoader
import pandas as pd
import logging
from datasets_loader import load_sweet_bitter_dataset, load_odor_dataset, load_common_tags_dataset
from lightning_module import GNNLightning
import seaborn as sns
import pickle

SAVE_ACTIVATIONS = True
ACTIVATIONS_PATH = '../results/activations/'
PREDICTIONS_PATH = '../results/predictions/'

logging.basicConfig(level=logging.INFO)

XS = 12
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
BIGGEST_SIZE = 20

pyplot.rc('font', size=BIGGER_SIZE)  # controls default text sizes
pyplot.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
pyplot.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
pyplot.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
pyplot.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
pyplot.rc('legend', fontsize=XS)  # legend fontsize
pyplot.rc('figure', titlesize=BIGGEST_SIZE)  # fontsize of the figure title


def get_models_predictions(models, test_loader):
    models_predictions = {}
    for model_name, model_path in models.items():
        classifier = GNNLightning.load_from_checkpoint(model_path)
        classifier.eval()

        y = []
        pred_probs = []
        neg_preds = []
        pos_preds = []

        iterator = iter(test_loader)
        for molecule in iterator:
            classes = molecule.y.squeeze(dim=-1)
            classes = [clazz.item() for clazz in classes]
            y = classes

            preds = classifier.get_probs_as_list(molecule)
            pred_probs = preds

        models_predictions['expected_labels'] = y
        models_predictions[model_name] = pred_probs

        for label, prob in zip(y, pred_probs):
            if label == 1:
                pos_preds.append(prob)
            else:
                neg_preds.append(prob)

    return models_predictions, y


def get_multilabel_models_predictions_2_logits(models, test_loader):
    models_predictions = {}
    y = None
    for model_name, model_path in models.items():
        classifier = GNNLightning.load_from_checkpoint(model_path)
        classifier.eval()

        y = []
        pred_probs = []
        neg_preds = []
        pos_preds = []

        iterator = iter(test_loader)
        for molecule in iterator:
            classes = molecule.y
            y = {}
            for clasx in classes:
                class_labels = classes[clasx]
                class_labels = [clazz.item() for clazz in class_labels]

                y[clasx] = class_labels
            # classes = [clazz.item() for clazz in classes]
            # y = classes

            model_preds = classifier(molecule)
            preds = {}
            for category in model_preds:
                category_preds = [clazz.item() for clazz in model_preds[category].argmax(dim=-1)]
                preds[category] = category_preds
            # pred_probs = preds

        models_predictions['expected_labels'] = y
        models_predictions[model_name] = preds

    return models_predictions, y


def get_multilabel_models_probs_1_logit(models, test_loader, save_activations=False, activations_path=ACTIVATIONS_PATH):
    models_predictions = {}
    y = None
    for model_name, model_path in models.items():
        classifier = GNNLightning.load_from_checkpoint(model_path)
        classifier.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classifier.to(device)

        y = {}
        pred_probs = []
        neg_preds = []
        pos_preds = []

        iterator = iter(test_loader)
        molecules_predictions = {}
        preds = {}
        for molecule in iterator:
            molecule = molecule.to(device)
            name = molecule.name
            molecules_predictions[name[0]] = {}
            classes = molecule.y

            for clasx in classes:
                class_labels = classes[clasx]
                class_labels = [clazz.item() for clazz in class_labels]
                if y.get(clasx) is not None:
                    y[clasx] = y[clasx] + class_labels
                else:
                    y[clasx] = class_labels
            # classes = [clazz.item() for clazz in classes]
            # y = classes

            model_preds = classifier(molecule)

            for category in model_preds:
                category_preds = torch.sigmoid(model_preds[category].squeeze(dim=-1))
                molecules_predictions[name[0]][category] = round(category_preds.item(), 5)
                if preds.get(category) is not None:
                    preds[category] = torch.cat((preds[category], category_preds), dim=0)
                else:
                    preds[category] = category_preds
            # pred_probs = preds
        classifier.save_activations(activations_path + f'/{model_name}')

        save_model_predictions(molecules_predictions, model_name)
        models_predictions['expected_labels'] = y
        models_predictions[model_name] = preds

    return models_predictions, y


def get_multilabel_logits_by_label(model_name, model_path, test_loader, activations_path=ACTIVATIONS_PATH):
    logging.info('Sigmoid will be applied to the logits in the analysis')
    classifier = GNNLightning.load_from_checkpoint(model_path)
    classifier.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.to(device)

    iterator = iter(test_loader)
    molecules_predictions = {}
    preds = {}
    for molecule in iterator:
        molecule = molecule.to(device)
        name = molecule.name
        molecules_predictions[name[0]] = {}

        model_preds = classifier(molecule)

        for category in model_preds:
            category_preds = torch.sigmoid(model_preds[category].squeeze(dim=-1))
            pred = round(category_preds.item(), 5)
            molecules_predictions[name[0]][category] = pred
            if preds.get(category) is not None:
                preds[category] = torch.cat((preds[category], category_preds), dim=0)
            else:
                preds[category] = category_preds

    classifier.save_activations(activations_path + f'/{model_name}')
    save_model_predictions(molecules_predictions, model_name)

    return molecules_predictions


def save_dict_to_csv(data, filename):
    import csv
    """
    Save a dictionary to a CSV file.

    :param data: Dictionary where keys are molecule names and values are dictionaries of category activations.
    :param filename: Name of the CSV file to save.
    """
    if not data:
        print("The provided data dictionary is empty.")
        return

    # Get the header from the keys of the first molecule
    header = ['molecule'] + list(next(iter(data.values())).keys())

    # Prepare rows
    rows = [[molecule] + [attributes[category] for category in header[1:]] for molecule, attributes in data.items()]

    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # Write header
        writer.writerows(rows)  # Write data rows

    print(f"Data has been successfully saved to {filename}.")


def save_model_predictions(preds, model_name):
    with open(PREDICTIONS_PATH + f'{model_name}.pickle', 'wb') as handle:
        pickle.dump(preds, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_models_metrics(models_predictions, y):
    models_metrics = {}
    labels = y

    for model_name, predictions in models_predictions.items():
        if model_name != 'expected_labels':
            model_fpr, model_tpr, model_threshs = roc_curve(y, predictions)
            model_auc = roc_auc_score(labels, predictions)
            models_metrics[model_name] = {}

            models_metrics[model_name]['fpr'] = model_fpr
            models_metrics[model_name]['tpr'] = model_tpr
            models_metrics[model_name]['auc'] = model_auc

            precision, recall, thresholds = precision_recall_curve(labels, predictions)
            models_metrics[model_name]['precision'] = precision
            models_metrics[model_name]['recall'] = recall

            models_metrics[model_name]['confusion_matrix'] = confusion_matrix(y, [round(x) for x in predictions])

    return models_metrics


def get_multilabel_models_metrics(models_predictions):
    models_metrics = {}

    for model_name, predictions in models_predictions.items():

        if model_name != 'expected_labels':
            if models_metrics.get(model_name) is None:
                models_metrics[model_name] = {}

            for category, category_preds in predictions.items():
                category_y = models_predictions['expected_labels'][category]
                models_metrics[model_name][f'{category}_confusion_matrix'] = confusion_matrix(category_y,
                                                                                              category_preds)

    return models_metrics


def plot_roc_multi(models_metrics, y, tags, model_name):
    ns_probs = [1 for _ in range(len(y[tags[0]]))]

    # plot random model
    ns_fpr, ns_tpr, _ = roc_curve(y[tags[0]], ns_probs)

    fig, axn = pyplot.subplots(1, len(models_metrics), sharex=False, sharey=False, constrained_layout=True)
    fig.suptitle(f'Roc Curves {model_name}')
    # fig.delaxes(axn[1][1])
    # fig.delaxes(axn[1][2])

    for ax, tag in zip(axn.flat, models_metrics):
        fpr = list(models_metrics[tag]['fpr'])
        tpr = list(models_metrics[tag]['tpr'])
        name_auc = [f'{tag.capitalize()} ' + '( AUC = ' + str(round(models_metrics[tag]['auc'], 2)) + ')' for x in
                    range(len(models_metrics[tag]['tpr']))]

        roc = pd.DataFrame.from_dict({'FPR': (fpr),  # + list(model_fpr2),
                                      'TPR': (tpr),  # + list(model_tpr2),
                                      '': name_auc})  #
        ax.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill', color='orange')

        sns.set_style("whitegrid")
        sns.lineplot(data=roc, x="FPR", y="TPR", hue='', ax=ax,
                     palette=sns.color_palette('bright')[3:4])

        ax.grid(True)

    pyplot.legend(loc='lower right')
    pyplot.legend()
    # fig.tight_layout()
    # show the plot
    pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    pyplot.show()
    # pyplot.savefig('roc_' + model_name + '.png')
    pyplot.close()


def plot_roc_multi_all_models_together_old(models_metrics, y, tags, model_name, save=True):
    ns_probs = [1 for _ in range(len(y[tags[0]]))]

    # plot random model
    ns_fpr, ns_tpr, _ = roc_curve(y[tags[0]], ns_probs)

    fig, axn = pyplot.subplots(2, 3, sharex=False, sharey=False, constrained_layout=True)
    fig.delaxes(axn[1][2])
    fig.suptitle(f'Roc Curves {model_name}')
    # fig.delaxes(axn[1][1])
    # fig.delaxes(axn[1][2])

    for ax, tag in zip(axn.flat, tags):
        ax.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill', color='black')
        color = 1
        for modell_metrics in models_metrics:
            model_metrics = models_metrics[modell_metrics]
            fpr = list(model_metrics[tag]['fpr'])
            tpr = list(model_metrics[tag]['tpr'])
            name_auc = [f'{modell_metrics}' + ': ' + str(round(model_metrics[tag]['auc'], 2)) for x in
                        range(len(model_metrics[tag]['tpr']))]

            roc = pd.DataFrame.from_dict({'FPR': (fpr),  # + list(model_fpr2),
                                          'TPR': (tpr),  # + list(model_tpr2),
                                          '': name_auc})  #

            sns.set_style("whitegrid")
            sns.lineplot(data=roc, x="FPR", y="TPR", hue='', ax=ax,
                         palette=sns.color_palette('bright')[color - 1:color])
            color += 1

        ax.set_title(tag.capitalize())
        ax.grid(True)

        ax.legend(loc='lower right', title='AUC')
        ax.get_legend().get_title().set_fontsize('12')

    # fig.tight_layout()
    # show the plot
    # pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    # pyplot.gcf().set_size_inches(20, 40)
    # pyplot.tight_layout()

    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'roc_{model_name}_{timestamp}.png'
        pyplot.savefig(filename)
    pyplot.show()
    pyplot.close()


def plot_roc_multi_all_models_together(models_metrics, y, tags, model_name, save=True):
    ns_probs = [1 for _ in range(len(y[tags[0]]))]

    # plot random model
    ns_fpr, ns_tpr, _ = roc_curve(y[tags[0]], ns_probs)

    fig, axn = pyplot.subplots(2, 3, figsize=(18, 12), sharex=False, sharey=False, constrained_layout=True)
    fig.delaxes(axn[1, 2])  # Remove the last subplot to accommodate 5 plots

    fig.suptitle(f'Roc Curves {model_name}')

    for ax, tag in zip(axn.flat, tags):
        ax.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill', color='black')
        color = 1
        for modell_metrics in models_metrics:
            model_metrics = models_metrics[modell_metrics]
            fpr = list(model_metrics[tag]['fpr'])
            tpr = list(model_metrics[tag]['tpr'])
            name_auc = [f'{modell_metrics}' + ': ' + str(round(model_metrics[tag]['auc'], 2)) for x in
                        range(len(model_metrics[tag]['tpr']))]

            roc = pd.DataFrame.from_dict({'FPR': (fpr), 'TPR': (tpr), '': name_auc})

            sns.set_style("whitegrid")
            sns.lineplot(data=roc, x="FPR", y="TPR", hue='', ax=ax,
                         palette=sns.color_palette('bright')[color - 1:color])
            color += 1

        ax.set_title(tag.capitalize())
        ax.grid(True)
        ax.legend(loc='lower right', title='AUC')
        ax.get_legend().get_title().set_fontsize('12')

    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'../results/roc-pr/roc_{model_name}_{timestamp}.png'
        pyplot.savefig(filename)
    pyplot.show()
    pyplot.close()


def plot_pr_multi_all_models_together(models_metrics, y, tags, model_name, save=True):
    ns_probs = [1 for _ in range(len(y[tags[0]]))]

    fig, axn = pyplot.subplots(2, 3, figsize=(18, 12), sharex=False, sharey=False, constrained_layout=True)
    fig.delaxes(axn[1, 2])  # Remove the last subplot to accommodate 5 plots

    fig.suptitle(f'Precision-Recall Curves {model_name}')

    for ax, tag in zip(axn.flat, tags):
        color = 1
        for modell_metrics in models_metrics:
            model_metrics = models_metrics[modell_metrics]
            precision = list(model_metrics[tag]['precision'])
            recall = list(model_metrics[tag]['recall'])

            prc = pd.DataFrame.from_dict({'Recall': recall,
                                          'Precision': precision,
                                          '': [modell_metrics for x in range(len(model_metrics[tag]['precision']))]})

            sns.set_style("whitegrid")
            sns.lineplot(data=prc, x="Recall", y="Precision", hue='', ax=ax,
                         palette=sns.color_palette('bright')[color - 1:color])
            color += 1

        ax.set_title(tag.capitalize())
        ax.grid(True)
        ax.legend(loc='lower right')

    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'../results/roc-pr/pr_{model_name}_{timestamp}.png'
        pyplot.savefig(filename)

    pyplot.show()
    pyplot.close()


def plot_pr_multi_all_models_together_old(models_metrics, y, tags, model_name):
    ns_probs = [1 for _ in range(len(y[tags[0]]))]

    # plot random model
    ns_fpr, ns_tpr, _ = roc_curve(y[tags[0]], ns_probs)

    fig, axn = pyplot.subplots(2, 3, sharex=False, sharey=False, constrained_layout=True)
    fig.delaxes(axn[1][2])
    fig.suptitle(f'Precision-Recall Curves {model_name}')
    # fig.delaxes(axn[1][1])
    # fig.delaxes(axn[1][2])

    for ax, tag in zip(axn.flat, tags):
        color = 1
        for modell_metrics in models_metrics:
            model_metrics = models_metrics[modell_metrics]
            precision = list(model_metrics[tag]['precision'])
            recall = list(model_metrics[tag]['recall'])

            roc = pd.DataFrame.from_dict({'Recall': recall,  # + list(model_fpr2),
                                          'Precision': precision,  # + list(model_tpr2),
                                          '': [modell_metrics for x in range(len(model_metrics[tag]['precision']))]})  #

            sns.set_style("whitegrid")
            sns.lineplot(data=roc, x="Recall", y="Precision", hue='', ax=ax,
                         palette=sns.color_palette('bright')[color - 1:color])
            color += 1

        ax.set_title(tag.capitalize())
        ax.grid(True)

        ax.legend(loc='lower right')
        # ax.get_legend().get_title().set_fontsize('12')

    # fig.tight_layout()
    # show the plot
    # pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    pyplot.show()
    # pyplot.savefig('roc_' + model_name + '.png')
    pyplot.close()


def plot_precission_recall_multi(models_metrics, y, tags, model_name):
    fig, axn = pyplot.subplots(1, len(models_metrics), sharex=False, sharey=False)
    fig.suptitle(f'Precision-Recall Curves {model_name}')

    for ax, tag in zip(axn.flat, models_metrics):
        pr = list(models_metrics[tag]['precision'])
        rc = list(models_metrics[tag]['recall'])
        tags = [f'{tag.capitalize()} ' for x in
                range(len(models_metrics[tag]['precision']))]

        roc = pd.DataFrame.from_dict({'Precision': pr, 'Recall': rc, '': tags})  #

        sns.set_style("whitegrid")
        sns.lineplot(data=roc, x="Recall", y="Precision", hue='', ax=ax, palette=sns.color_palette('bright')[:1])
        # ax.set_title(tag.capitalize())
        ax.grid(True)
        ax.legend(loc='lower left')

    # pyplot.legend(loc='lower left')
    # pyplot.legend()
    # show the plot
    pyplot.show()
    pyplot.close()


def plot_precision_recall(models_metrics, key='Model', amt_of_models=1):
    ns_probs = [0 for _ in range(len(y))]

    pyplot.suptitle('Precision-Recall curve')

    recall = []
    precision = []
    name_auc = []

    for model_name, models_predictions in models_metrics.items():
        recall += list(models_predictions['recall'])
        precision += list(models_predictions['precision'])
        name_auc += [f'{model_name}' for x in
                     range(len(models_predictions['recall']))]

    roc = pd.DataFrame.from_dict({'Recall': (recall),  # + list(model_fpr2),
                                  'Precision': (precision),  # + list(model_tpr2),
                                  'Model': name_auc})  #
    sns.set_style("whitegrid")
    ax = sns.lineplot(data=roc, x="Recall", y="Precision", hue='Model',
                      palette=sns.color_palette('bright')[:len(models_metrics)])
    ax.grid(True)

    pyplot.legend(loc='lower right')
    pyplot.legend()
    # show the plot
    pyplot.show()
    pyplot.close()


def plot_scores_distribution(model_name, pred_probs, y):
    scores = pd.DataFrame.from_dict({'Score': pred_probs, 'Predicted Label': y})
    sns.displot(data=scores, kind='kde', x='Score', hue='Predicted Label', fill=True,
                palette=sns.color_palette('bright')[:2], aspect=1.5)

    pyplot.suptitle(f'Scores distribution ({model_name})')
    pyplot.legend()
    pyplot.show()
    pyplot.close()


def plot_confusion_matrix(model_name, confusion_ndarray):
    df_cm = pd.DataFrame(confusion_ndarray, index=[i for i in "01"],
                         columns=[i for i in "01"])
    pyplot.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt="d")
    pyplot.suptitle(f'Confusion Matrix ({model_name})')
    pyplot.show()
    pyplot.close()


def plot_confusion_matrix_multilabel(model_name, confusion_ndarray):
    fig, axn = pyplot.subplots(2, 3, sharex=True, sharey=True)
    fig.suptitle(f'Confusion Matrix ({model_name})')

    for ax, category in zip(axn.flat, confusion_ndarray):
        # sns.heatmap(df, ax=ax)
        df_cm = pd.DataFrame(confusion_ndarray[category], index=[i for i in "01"],
                             columns=[i for i in "01"])
        cmap = {
            'Green': 'Greens',
            'Floral': 'PuRd',
            'Woody': 'Oranges',
            'Fruity': 'Wistia',
            'Bitter': 'Blues'
        }
        name = category.split('_')[0].capitalize()
        sns.heatmap(df_cm, annot=True, ax=ax, fmt="d", cmap=cmap[name])
        ax.set_title(name)

    fig.delaxes(axn[1, 2])
    pyplot.show()
    pyplot.close()


def plot_scores_distributions_same_plot(model_predictions_, y):
    # dos filas dos columnas

    fig, axn = pyplot.subplots(2, 2, sharex=False, sharey=False)
    fig.suptitle(f'Scores distribution')

    model_predictions = deepcopy(model_predictions_)
    del model_predictions['expected_labels']
    for ax, tag in zip(axn.flat, model_predictions):
        scores = pd.DataFrame.from_dict({'Score': model_predictions[model], 'Predicted Label': y})

        sns.kdeplot(data=scores, x='Score', hue='Predicted Label', fill=True,
                    palette=sns.color_palette('bright')[:2], ax=ax, thresh=1)
        ax.set_title(model)

    # pyplot.legend()
    pyplot.show()
    pyplot.close()


def multilabel_metrics(models, models_predictions, tags=('fruity', 'bitter', 'green', 'floral', 'woody'), y=None):
    model_metrics = {}
    for model in models:
        model_metrics[model] = {}
        for tag in tags:
            # for model in models:
            model_metrics[model][tag] = {}

            precision, recall, thresholds = precision_recall_curve(y[tag], models_predictions[model][tag].tolist())

            model_metrics[model][tag]['precision'] = precision
            model_metrics[model][tag]['recall'] = recall

            model_fpr, model_tpr, model_threshs = roc_curve(y[tag], models_predictions[model][tag].tolist())
            model_auc = roc_auc_score(y[tag], models_predictions[model][tag].tolist())

            model_metrics[model][tag]['fpr'] = model_fpr
            model_metrics[model][tag]['tpr'] = model_tpr
            model_metrics[model][tag]['auc'] = model_auc

    return model_metrics


def save_roc_pr_images_for_models(models, test_loader, tags):
    models_predictions, y = get_multilabel_models_probs_1_logit(models, test_loader, SAVE_ACTIVATIONS)
    model_metrics = multilabel_metrics(models, models_predictions, y=y)
    plot_roc_multi_all_models_together(model_metrics, y, tags, model_name='', save=True)
    plot_pr_multi_all_models_together(model_metrics, y, tags, model_name='', save=True)


def save_labels_logits(models, test_loader):
    for model_name, model in models.items():
        models_predictions = get_multilabel_logits_by_label(model_name, model, test_loader)
        save_dict_to_csv(models_predictions, '../results/label-logits/' + model_name + '_label_logits' + '.csv')
        for x, v in models_predictions.items():
            print(x, v)


if __name__ == '__main__':
    GAT_0 = '/home/tomas/PycharmProjects/odors/experiments/three_layers_retrain/lightning_logs/version_105/checkpoints/epoch=54-step=2145.ckpt'
    GAT_1 = '/home/tomas/PycharmProjects/odors/experiments/three_layers_retrain/lightning_logs/version_143/checkpoints/epoch=21-step=858.ckpt'
    GAT_2 = '/home/tomas/PycharmProjects/odors/experiments/three_layers_retrain/lightning_logs/version_53/checkpoints/epoch=45-step=1794.ckpt'
    GAT_3 = '/home/tomas/PycharmProjects/odors/experiments/three_layers_retrain/lightning_logs/version_31/checkpoints/epoch=15-step=624.ckpt'

    # despues es menor test lost
    # GCN = '../experiments/common_tags_balanced_4/lightning_logs/version_131/checkpoints/epoch=59-step=1919.ckpt'
    # MLP = '../experiments/common_tags_balanced_4/lightning_logs/version_68/checkpoints/epoch=14-step=959.ckpt'
    # ATTENTIVE = '../experiments/common_tags_balanced_4/lightning_logs/version_117/checkpoints/epoch=71-step=1511.ckpt'

    tags = ('fruity', 'bitter', 'green', 'floral', 'woody')

    # train, validation, test, weights = load_common_tags_dataset(tags=tags,
    #                                                            dataset_path='../dataset/common_tags/unbalanced',
    #                                                            test_set_path='/home/tomas/PycharmProjects/tesis/dataset/common_tags/colaboracion_activaciones/enzo_eric_mols.csv',
    #                                                            bypas_intersections=True,
    #                                                            show_tags=False)

    multilabel_tags = ['green', 'bitter', 'fruity', 'floral', 'woody']
    train, validation, test, weights = load_common_tags_dataset(tags=multilabel_tags,
                                                                dataset_path='../dataset/common_tags/unbalanced')

    batch_size_test = 1
    batch_size_validation = 1

    test_loader = DataLoader(test, batch_size=batch_size_test, shuffle=False, drop_last=False)

    models = {
        'GAT_0': GAT_0,
        'GAT_1': GAT_1,
        'GAT_2': GAT_2,
        'GAT_3': GAT_3,
    }

    models_predictions, y = get_multilabel_models_probs_1_logit(models, test_loader, SAVE_ACTIVATIONS)

    model_metrics = multilabel_metrics(models, models_predictions, y)
    plot_roc_multi_all_models_together(model_metrics, y, tags, model_name='', save=True)
    plot_pr_multi_all_models_together(model_metrics, y, tags, model_name='', save=True)

    # models_predictions_val, y_val = get_multilabel_models_probs_1_logit(models, validation_loader)
