from torch_geometric.data import DataLoader

from model_analysis import load_common_tags_dataset, save_dict_to_csv, get_multilabel_logits_by_label, \
    get_multilabel_models_probs_1_logit, multilabel_metrics, plot_roc_multi_all_models_together, \
    plot_pr_multi_all_models_together, save_labels_logits, save_roc_pr_images_for_models
from modules.datasets_loader import load_common_tags_dataset_no_split

GAT_0 = '/home/tomas/PycharmProjects/odors/experiments/three_layers_retrain/lightning_logs/version_105/checkpoints/epoch=54-step=2145.ckpt'
GAT_1 = '/home/tomas/PycharmProjects/odors/experiments/three_layers_retrain/lightning_logs/version_143/checkpoints/epoch=21-step=858.ckpt'
modelo_3 = '/home/tomas/PycharmProjects/odors/experiments/three_layers_retrain/lightning_logs/version_53/checkpoints/epoch=45-step=1794.ckpt'
modelo_2 = '/home/tomas/PycharmProjects/odors/experiments/test_same_dimension/lightning_logs/version_9/checkpoints/epoch=24-step=2925.ckpt'
modelo_1 = '/home/tomas/PycharmProjects/odors/experiments/test_same_dimension/lightning_logs/version_8/checkpoints/epoch=3-step=232.ckpt'
tags = ('fruity', 'bitter', 'green', 'floral', 'woody')

multilabel_tags = ['green', 'bitter', 'fruity', 'floral', 'woody']
train, validation, test, weights = load_common_tags_dataset(tags=multilabel_tags,
                                                            dataset_path='../dataset/common_tags/unbalanced')

test_mols = load_common_tags_dataset_no_split(
    '/home/tomas/PycharmProjects/odors/dataset/common_tags/molecules_inchi.csv')

batch_size_test = 1

test_loader = DataLoader(test, batch_size=batch_size_test, shuffle=False, drop_last=False)
test_loader_eric = DataLoader(test_mols, batch_size=batch_size_test, shuffle=False, drop_last=False)

models = {
    'modelo_1': modelo_1,
    'modelo_2': modelo_2
}

SAVE_ACTIVATIONS = True

# Guardamos las logits de cada clase para cada modelo (no se necesita tener etiquetado el datadaset)
save_labels_logits(models, test_loader_eric)

# Guardamos las curvas ROC y PR con los labels del dataset (solo usar si tenemos los labels puestos)
save_roc_pr_images_for_models(models, test_loader, tags)

# models_predictions_val, y_val = get_multilabel_models_probs_1_logit(models, validation_loader)
