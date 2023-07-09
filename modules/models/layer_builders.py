from torch.nn import BatchNorm1d, Sequential, Dropout, ModuleList, Linear, ReLU
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, GATConv, GraphConv, global_add_pool

activations_by_name = {
    'relu': ReLU()
}


def build_convolutional_module(hp):
    gnn_layer_by_name = {
        "GCN": GCNConv,
        "GATN": GATConv,
        "GraphConv": GraphConv
    }

    batch_normalization = hp['batch_normalization']

    layers = []
    convolutions = hp['convolutions']
    for conv in convolutions:
        conv_type = conv['conv_type']

        conv_module = gnn_layer_by_name[conv_type]
        in_features = conv['in_features']
        out_channels = conv['out_channels']

        edge_dim = conv.get('edge_dim')
        if edge_dim is None or edge_dim == 0:
            convolution = conv_module(in_channels=in_features, out_channels=out_channels)
        else:
            convolution = conv_module(in_channels=in_features, out_channels=out_channels, edge_dim=edge_dim)
        dropout = conv['dropout']
        activation = activations_by_name[conv['activation']]

        final_conv = [convolution, activation]

        if batch_normalization:
            final_conv.append(BatchNorm1d(out_channels))
        if dropout > 0:
            final_conv.append(Dropout(dropout))
        layers += final_conv

    layers = ModuleList(layers)
    return layers


def build_graph_pool_layer(hp):
    pool_layer_by_name = {
        "add": global_add_pool,
        "max": global_max_pool,
        "mean": global_mean_pool
    }

    return pool_layer_by_name[hp['pool_type']]


def build_affine_layers(hp, one_logit_ouput=True):
    number_of_affine_hidden_layers = hp['hidden_layers_number']

    if number_of_affine_hidden_layers < 0 or number_of_affine_hidden_layers > 4:
        raise ('Invalid number of affine layers, must be a value between 2 and 4')

    affine_hidden_sizes = hp['affine_hidden_layers']

    activation = activations_by_name[hp['activation']]
    batch_normalization = hp['batch_normalization']
    # tomamos los out channels la ultima convolucion definida
    if hp['model'] == 'MLP' or hp['model'] == 'MLPMultilabel':
        number_of_input_features = hp['number_of_features_input_layer']
    else:
        number_of_input_features = hp['convolutions'][-1]['out_channels']
    dropout = hp['dropout']

    if number_of_affine_hidden_layers == 0:
        # si no hay hidden layers terminamos con un unico logit
        hidden_layers = [Linear(number_of_input_features, 1)]
    else:
        hidden_layers = [Linear(number_of_input_features, affine_hidden_sizes[0]), activation]
        if batch_normalization:
            hidden_layers.append(BatchNorm1d(affine_hidden_sizes[0]))

        for index in range(len(affine_hidden_sizes) - 1):
            hidden_layers.append(Linear(affine_hidden_sizes[index], affine_hidden_sizes[index + 1]))
            hidden_layers.append(activation)
            if batch_normalization:
                hidden_layers.append(BatchNorm1d(affine_hidden_sizes[index + 1]))
            if dropout > 0:
                hidden_layers.append(Dropout(dropout))
        if one_logit_ouput:
            hidden_layers.append(Linear(affine_hidden_sizes[-1], 1))

    # save affine layers in hparams
    layers_structure = []
    for index, layer in enumerate(hidden_layers):
        if isinstance(layer, Linear):
            layers_structure.append({'layer_' + str(index): {'in_features': layer.in_features,
                                                             'out_features': layer.out_features}})
        else:
            layers_structure.append({'layer_' + str(index): type(layer).__name__})

            hp['affine_layers'] = layers_structure
            hidden_layers = Sequential(*hidden_layers)

            # save affine layers in hparams

    return hidden_layers
