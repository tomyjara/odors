import torch
from torch import sigmoid

from torch_geometric.nn import GATConv
from torch_geometric.nn import global_add_pool, MessagePassing

from pprint import pprint

from modules.models.layer_builders import build_convolutional_module, build_graph_pool_layer, build_affine_layers



from torch_geometric.nn.models import AttentiveFP


class Model(torch.nn.Module):
    @staticmethod
    def is_multilabel():
        return False
class GCN(Model):
    def __init__(self, hp):
        super(GCN, self).__init__()

        self.convs = build_convolutional_module(hp)
        self.conv_pool = build_graph_pool_layer(hp)
        self.fully_connected = build_affine_layers(hp)
        self.apply_sigmoid_in_last_layer = hp['apply_sigmoid_in_last_layer'] if hp.get(
            'apply_sigmoid_in_last_layer') is not None else False

    def forward(self, data):
        x, edge_index, edge_attributes = data.x, data.edge_index, data.edge_attr
        for layer in self.convs:
            if isinstance(layer, MessagePassing):
                if isinstance(layer, GATConv) and layer.edge_dim is not None:
                    x = layer(x, edge_index, edge_attributes)
                else:
                    try:
                        x = layer(x, edge_index)
                    except Exception as e:
                        print(e)
            else:
                x = layer(x)

        x = self.conv_pool(x, data.batch)
        x = self.fully_connected(x)
        # x = F.log_softmax(x, dim=1)
        # x = sigmoid(x)
        if self.apply_sigmoid_in_last_layer:
            return sigmoid(x)
        else:
            return x


class MLP(Model):
    def __init__(self, hp):
        super(MLP, self).__init__()

        self.fully_connected = build_affine_layers(hp)
        self.apply_sigmoid_in_last_layer = hp['apply_sigmoid_in_last_layer'] if hp.get(
            'apply_sigmoid_in_last_layer') is not None else False

    def forward(self, data):
        x, edge_index, edge_attributes = data.x, data.edge_index, data.edge_attr
        x = global_add_pool(x, data.batch)
        x = self.fully_connected(x)

        # x = F.log_softmax(x, dim=1)
        # x = sigmoid(x)
        if self.apply_sigmoid_in_last_layer:
            return sigmoid(x)
        else:
            return x


class AttentiveFp(Model):
    def __init__(self, hp):
        super(AttentiveFp, self).__init__()

        attentive_hp = hp['attentive_fp']

        input_features = attentive_hp['input_features']
        hidden_channels = attentive_hp['hidden_channels']
        out_channels = attentive_hp['out_channels']
        edge_dim = attentive_hp['edge_dim']
        num_layers = attentive_hp['num_layers']
        num_timesteps = attentive_hp['num_timesteps']
        dropout = attentive_hp['dropout']

        self.apply_sigmoid_in_last_layer = hp['apply_sigmoid_in_last_layer'] if hp.get(
            'apply_sigmoid_in_last_layer') is not None else False

        self.model = AttentiveFP(in_channels=input_features,
                                 hidden_channels=hidden_channels,
                                 out_channels=out_channels,
                                 edge_dim=edge_dim,
                                 num_layers=num_layers,
                                 num_timesteps=num_timesteps,
                                 dropout=dropout)

    def forward(self, data):
        x, edge_index, edge_attributes = data.x, data.edge_index, data.edge_attr
        x = self.model(x, edge_index, edge_attributes, data.batch)

        # x = F.log_softmax(x, dim=1)
        # x = sigmoid(x)
        if self.apply_sigmoid_in_last_layer:
            return sigmoid(x)
        else:
            return x
