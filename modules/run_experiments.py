import pandas as pd
import os

from pprint import pprint
from torch_geometric.loader import DataLoader
from modules.datasets_loader import load_common_tags_dataset

from modules.hyperparams import select_random_hyperparams
from modules.trainer import train_graph_classifier


EXP_NAME = 'test_same_dimension'
NUMBER_OF_EXPERIMENTS = 5

if __name__ == '__main__':
    # pl.seed_everything(42)

    exp_name = EXP_NAME
    CHECKPOINT_PATH = f"../experiments/{exp_name}"
    ROOT_DIR = os.path.abspath(os.path.join(CHECKPOINT_PATH))

    print(('Launching ' + str(NUMBER_OF_EXPERIMENTS) + ' experiments'))

    exp_results = {
        'exp_number': [],
        # 'accuracy': [],
        # 'precision': [],
        # 'recall': [],
        # 'f1_score': [],
        # 'best_thresh': [],
        'model': []
    }

    #train2, validation2, test2, weights2 = load_common_tags_dataset(tags=multilabel_tags, dataset_path='../dataset/common_tags/balanced_tags_5')
    #train, validation, test = load_sulfur_floral_dataset()
    #train, validation, test = load_sweet_bitter_dataset()

    multilabel_tags = ['green', 'bitter', 'fruity', 'floral', 'woody']
    train, validation, test, weights = load_common_tags_dataset(tags=multilabel_tags, dataset_path='../dataset/common_tags/unbalanced')

    for x in range(NUMBER_OF_EXPERIMENTS):
        hyperparams = select_random_hyperparams(exp_name=EXP_NAME)
        hyperparams['multilabel_tags'] = multilabel_tags
        # Esto es por si se quieren usar los pesos
        hyperparams['category_weights'] = None

        print('Model: ' + str(hyperparams['model']))
        exp_name = hyperparams['exp_name']

        batch_size = hyperparams['batch_size']

        train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test, batch_size=len(test), shuffle=False, drop_last=False)
        validation_loader = DataLoader(validation, batch_size=len(validation), shuffle=False, drop_last=False)

        best_model, result = train_graph_classifier(hyperparams, ROOT_DIR, train_loader, test_loader, validation_loader)

        exp_results['exp_number'].append(x)
        #exp_results['model'].append(result[0]['model'])

        for metric in result[0]:
            if exp_results.get(metric) is None:
                exp_results[metric] = [result[0][metric]]
            else:
                exp_results[metric].append(result[0][metric])

        print('Experiment ' + str(x) + ' completed')

    print('Experiment results')
    pprint(exp_results)
    results = pd.DataFrame(exp_results)

    results.to_csv(f'../results/{EXP_NAME}.csv', index=False)
