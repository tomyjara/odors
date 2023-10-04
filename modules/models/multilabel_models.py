from torch import sigmoid

from torch.nn import Linear, Module

from torch_geometric.nn import GATConv, AttentiveFP
from torch_geometric.nn import global_add_pool, MessagePassing, global_mean_pool

from modules.models.models import build_convolutional_module, build_graph_pool_layer, build_affine_layers

import pickle
import logging
logging.basicConfig(level=logging.INFO)

class MutilabelModel(Module):
    def __init__(self, hp):
        super(MutilabelModel, self).__init__()
        self.fully_connected = build_affine_layers(hp, one_logit_ouput=False)
        self.last_layer_out_channels = self.last_layer_out_channels(hp)

        self.labels = hp.get('multilabel_tags', ['fruity', 'floral', 'green', 'woody', 'bitter', 'herbal'])
        self.hp = hp

        self.apply_sigmoid_in_last_layer = hp['apply_sigmoid_in_last_layer'] if hp.get(
            'apply_sigmoid_in_last_layer') is not None else False
        #self.save_convolutions_activations = hp.get('save_conv_activations', False)
        self.activations = {}

        # assign ouput logits dinamically based on the required tags
        for tag in self.labels:
            self.__setattr__(tag, Linear(in_features=self.last_layer_out_channels, out_features=1))

    # A veces la ultima capa no es una capa lineal
    def last_layer_out_channels(self, hp):
        last_layer = self.fully_connected[-1]
        if not isinstance(last_layer, Linear):
            last_layer = self.fully_connected[-2]
            if not isinstance(last_layer, Linear):
                last_layer = self.fully_connected[-3]
                if not isinstance(last_layer, Linear):
                    last_layer = self.fully_connected[-4]
        last_layer_out_channels = last_layer.out_features
        return last_layer_out_channels

    def build_multilabel_output(self, x):
        result = {}

        # apply last layer for each tag
        for tag in self.labels:
            prediction = self.__getattr__(tag)(x)
            if self.apply_sigmoid_in_last_layer:
                result[tag] = sigmoid(prediction)
            else:
                result[tag] = prediction

        return result

    def save_activations(self, out_path):
        with open(out_path + '.pickle', 'wb') as handle:
            pickle.dump(self.activations, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def is_multilabel():
        return True


class AttentiveFpMultilabel(MutilabelModel):
    def __init__(self, hp):
        super(AttentiveFpMultilabel, self).__init__(hp)
        attentive_hp = hp['attentive_fp']

        input_features = attentive_hp['input_features']
        hidden_channels = attentive_hp['hidden_channels']
        out_channels = attentive_hp['out_channels']
        edge_dim = attentive_hp['edge_dim']
        num_layers = attentive_hp['num_layers']
        num_timesteps = attentive_hp['num_timesteps']
        dropout = attentive_hp['dropout']

        self.model = AttentiveFP(in_channels=input_features,
                                 hidden_channels=hidden_channels,
                                 out_channels=out_channels,
                                 edge_dim=edge_dim,
                                 num_layers=num_layers,
                                 num_timesteps=num_timesteps,
                                 dropout=dropout)

    def last_layer_out_channels(self, hp):
        attentive_hp = hp['attentive_fp']
        out_channels = attentive_hp['out_channels']

        return out_channels

    def forward(self, data):
        x, edge_index, edge_attributes = data.x, data.edge_index, data.edge_attr
        x = self.model(x, edge_index, edge_attributes, data.batch)

        # x = F.log_softmax(x, dim=1)
        # x = sigmoid(x)
        result = self.build_multilabel_output(x)

        return result


class GCNMultilabel(MutilabelModel):
    def __init__(self, hp):
        super(GCNMultilabel, self).__init__(hp)
        self.convs = build_convolutional_module(hp)
        self.conv_pool = build_graph_pool_layer(hp)


    def forward(self, data):
        x, edge_index, edge_attributes, molecule_name = data.x, data.edge_index, data.edge_attr, data.name
        if molecule_name:
            self.activations[molecule_name[0]] = {}
        for layer in self.convs:
            if isinstance(layer, MessagePassing):
                if isinstance(layer, GATConv) and layer.edge_dim is not None:
                    x = layer(x, edge_index, edge_attributes)
                else:
                    x = layer(x, edge_index)
                if molecule_name:
                    pooled_activations = global_mean_pool(x, data.batch)
                    self.activations[molecule_name[0]][layer] = pooled_activations.flatten().tolist()
                    #self.activations[molecule_name[0]][layer] = x.flatten().tolist()
            else:
                x = layer(x)

        x = self.conv_pool(x, data.batch)
        x = self.fully_connected(x)
        # x = F.log_softmax(x, dim=1)
        # x = sigmoid(x)

        result = self.build_multilabel_output(x)

        return result


class MLPMultilabel(MutilabelModel):
    def __init__(self, hp):
        super(MLPMultilabel, self).__init__(hp)

    def forward(self, data):
        x, edge_index, edge_attributes = data.x, data.edge_index, data.edge_attr
        x = global_add_pool(x, data.batch)
        x = self.fully_connected(x)

        # x = F.log_softmax(x, dim=1)
        # x = sigmoid(x)
        result = self.build_multilabel_output(x)

        return result
