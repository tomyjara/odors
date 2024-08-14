import math
import random
from random import choice

NUMBER_OF_INPUT_FEATURES = 96
MAX_ALLOWED_AFFINE_LAYERS = 4


def define_hidden_layers(hp, layers):
    '''
    Esta funcion define la cantidad y la dimension de salida de las capas ocultas luego del modulo convolucional.
    Automaticamente se fija la cantidad de canales de salida del modulo convolucional y define
    posibles dimensiones para las capas a utilizar que van desde 8 a 1+log_2(#_canales_de_salida_del_modulo_convolucional).
    Luego selecciona aleatoriamente un numero de capas equivalente al especificado en hp y define la arquitectura en orden decreciente.

    Por ejemplo, si hp['hidden_layers_number'] = 3 y la ultima convolucion genera 128 canales, las posibles
    dimensiones estaran comprendidas entre [8, 16, 32, 64, 128] y se van a seleccionar 3 de modo
    que si las dimensiones elegidas resultan ser [16, 32, 8] el modulo fully connected nos quedaria con tres capas
    de dimensiones [32, 16, 8].

    Si se desea utilizar una cantidad de capas fijas con una dimension especifica se debe usar el parametro layers
    con la dimension salida de cada capa oculta. Por ejemplo pasandole a la funcion: layers=[8, 16, 4]
    '''
    if not layers:
        convolutions_out_channels = hp['convolutions'][-1]['out_channels']
        affine_possible_hidden_sizes = [2 ** x for x in range(3, int(math.log(convolutions_out_channels, 2)) + 1)]

        number_of_affine_hidden_layers = hp['hidden_layers_number']

        if number_of_affine_hidden_layers < 0 or number_of_affine_hidden_layers > MAX_ALLOWED_AFFINE_LAYERS:
            raise f'Invalid number of affine layers, must be a value between 1 and {MAX_ALLOWED_AFFINE_LAYERS}'

        affine_hidden_sizes = random.sample(affine_possible_hidden_sizes, number_of_affine_hidden_layers)
        affine_hidden_sizes = sorted(affine_hidden_sizes, reverse=True)

        hp['affine_hidden_layers'] = affine_hidden_sizes
    else:
        hp['affine_hidden_layers'] = layers

    return hp


def select_random_hyperparams(exp_name):
    hyperparams = {
        'exp_name': exp_name,
        'model': choice(['GCNMultilabel']), #, 'MLPMultilabel']),
        'apply_sigmoid_in_last_layer': False,
        'multilabel_tags': ['green', 'bitter', 'fruity', 'floral', 'woody'],
        'category_weights': None, # Si queremos especificar pesos para las distintas clases, no se si anda la verdad
        'weight_decay': choice([0.001, 0.01, 0.1, 0]),
        'learning_rate': choice([0.01, 0.02, 0.03]),
        'number_of_features_input_layer': NUMBER_OF_INPUT_FEATURES,
        'loss_module': 'BCEWithLogitsLoss',
        'optimizer': choice(['adamW', 'adam']),  # , 'adam', 'sdg']),
        'dropout': choice([0.1, 0.2, 0]),
        'max_epochs': 10000,
        'checkpoint_metric': 'val_loss', # La metrica que usamos para guardar los checkpoints de los modelos
        'early_stopping_min_delta': choice([0.001, 0.002]), # Cuanto consideramos que es una mejora de la metrica anterior
        'early_stopping_patience': choice([5, 10, 15]), # Cuantos epoochs esperamos para cortar el entrenamiento si la checkpoint_metric no mejora
        'batch_size': choice([10, 20, 30]),
        'pool_type': choice(['add', 'max']), # Pooling del modulo convolucional
        'batch_normalization': choice([True]),
        'hidden_layers_number': choice([1,2]), # Cantidad de hidden layers del modelo, la dimension de salida es aleatoria
        'fully_connected_manual_layers': None, # Salvo que la definas mediante una lista en este parametro, ej [32, 16, 8]
        'activation': choice(['relu']),
        'threshold': None,
        # Solo si se utiliza un modelo attentive fp, la verdad estos modelos no se si valen la pena ni siquiera
        'attentive_fp': {
            'input_features': NUMBER_OF_INPUT_FEATURES,
            'hidden_channels': choice([100, 150, 200]),
            'out_channels': choice([8, 16, 32]),
            'edge_dim': 12,
            'num_layers': choice([1, 2]),
            'num_timesteps': choice([1, 2, 3]),
            'dropout': choice([0, 0.2])

        },
        # Aca definimos las posibles arquitecturas de modulos convolucionales a experimentar, si queres definir
        # Una arquitectura fija la lista debe contener una sola entrada con esa arquitectura y los input y output
        # sizes tiene que matchear entre capas, la primera usa los input features
        'convolutions': choice([
            # [
            #     # ONE LAYER 64 CHANNELS
            #     {
            #         'conv_type': 'GCN',
            #         'in_features': NUMBER_OF_INPUT_FEATURES,
            #         'out_channels': 64,
            #         'edge_dim': 0,
            #         'dropout': 0,
            #         'activation': 'relu',
            #     }
            # ],
            #
            # [
            #     {
            #         'conv_type': 'GATN',
            #         'in_features': NUMBER_OF_INPUT_FEATURES,
            #         'out_channels': 64,
            #         'edge_dim': 12,
            #         'dropout': 0,
            #         'activation': 'relu',
            #     }
            # ],
            # [
            #     # ONE LAYER 96 CHANNELS
            #     {
            #         'conv_type': 'GCN',
            #         'in_features': NUMBER_OF_INPUT_FEATURES,
            #         'out_channels': 96,
            #         'edge_dim': 0,
            #         'dropout': 0,
            #         'activation': 'relu',
            #     }
            # ],
            #
            # [
            #     {
            #         'conv_type': 'GATN',
            #         'in_features': NUMBER_OF_INPUT_FEATURES,
            #         'out_channels': 96,
            #         'edge_dim': 12,
            #         'dropout': 0,
            #         'activation': 'relu',
            #     }
            # ],
            # # ONE LAYER 128 CHANNELS
            # [
            #     {
            #         'conv_type': 'GCN',
            #         'in_features': NUMBER_OF_INPUT_FEATURES,
            #         'out_channels': 128,
            #         'edge_dim': 0,
            #         'dropout': 0,
            #         'activation': 'relu',
            #     }
            # ],
            #
            # [
            #     {
            #         'conv_type': 'GATN',
            #         'in_features': NUMBER_OF_INPUT_FEATURES,
            #         'out_channels': 128,
            #         'edge_dim': 12,
            #         'dropout': 0,
            #         'activation': 'relu',
            #     }
            # ],
            #
            # # TWO LAYERS 64 CHANNELS
            # [
            #     {
            #         'conv_type': 'GCN',
            #         'in_features': NUMBER_OF_INPUT_FEATURES,
            #         'out_channels': 64,
            #         'edge_dim': 0,
            #         'dropout': 0,
            #         'activation': 'relu',
            #     },
            #
            #     {
            #         'conv_type': 'GCN',
            #         'in_features': 64,
            #         'out_channels': 64,
            #         'edge_dim': 0,
            #         'dropout': 0,
            #         'activation': 'relu',
            #     }
            # ],
            #
            # [
            #     {
            #         'conv_type': 'GATN',
            #         'in_features': NUMBER_OF_INPUT_FEATURES,
            #         'out_channels': 64,
            #         'edge_dim': 12,
            #         'dropout': 0,
            #         'activation': 'relu',
            #     },
            #
            #     {
            #         'conv_type': 'GATN',
            #         'in_features': 64,
            #         'out_channels': 64,
            #         'edge_dim': 12,
            #         'dropout': 0,
            #         'activation': 'relu',
            #     }
            # ],

            # # THREE LAYERS 32-64-32 CHANNELS
            [
                {
                    'conv_type': 'GATN',
                    'in_features': NUMBER_OF_INPUT_FEATURES,
                    'out_channels': 128,
                    'edge_dim': 0,
                    'dropout': 0,
                    'activation': 'relu',
                },

                {
                    'conv_type': 'GATN',
                    'in_features': 128,
                    'out_channels': 128,
                    'edge_dim': 0,
                    'dropout': 0,
                    'activation': 'relu',
                },
                {
                    'conv_type': 'GATN',
                    'in_features': 128,
                    'out_channels': 128,
                    'edge_dim': 0,
                    'dropout': 0,
                    'activation': 'relu',
                }
            ]
        ]),
    }

    fully_connected_defined_by_user = hyperparams['fully_connected_manual_layers']
    hyperparams = define_hidden_layers(hyperparams, fully_connected_defined_by_user)

    return hyperparams
