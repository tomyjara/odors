import pytorch_lightning as pl
import torch
import torch.optim as optim
import numpy as np
import logging

from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve
from sklearn.metrics._classification import f1_score
from torch.nn import BCELoss

from modules.models.models import AttentiveFp, GCN, MLP
from modules.models.multilabel_models import GCNMultilabel, MLPMultilabel, AttentiveFpMultilabel

logging.basicConfig(level=logging.INFO)
class GNNLightning(pl.LightningModule):

    def __init__(self, model_hparams):
        super().__init__()
        self.model_hparams = model_hparams
        self.model_name = model_hparams['model']

        self.model = self.initialize_model(model_hparams)
        # save hyperparams after model initialization in order to log affine layers structure
        # del model_hparams['affine_hidden_layers']
        self.save_hyperparameters()
        self.loss_module = self.initizlize_loss_module(model_hparams)
        self.category_weights = model_hparams.get('category_weights')

        if model_hparams.get('threshold') is not None:
            self.threshold = model_hparams.get('threshold')
            logging.info(f'Threshold set to {self.threshold }')
        else:
            self.threshold = 0.5
            logging.info('Threshold set to 0.5')

    def get_threshold(self):
        return self.threshold

    def initizlize_loss_module(self, model_hparams):
        weights = torch.tensor(np.array([2]), dtype=torch.long)
        loss_modules = {
            'NLLLoss': torch.nn.NLLLoss(),
            # 'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss(pos_weight=weights),
            'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss(),
            'BCELoss': BCELoss(),
            'CrossEntropyLoss': torch.nn.CrossEntropyLoss()

        }
        loss_module_name = model_hparams['loss_module']
        loss_module = loss_modules[loss_module_name]

        return loss_module

    def initialize_model(self, model_hparams):
        enabled_models = {
            'GCN': GCN,
            'MLP': MLP,
            'ATTENTIVE': AttentiveFp,
            'GCNMultilabel': GCNMultilabel,
            'MLPMultilabel': MLPMultilabel,
            'ATTENTIVEMultilabel': AttentiveFpMultilabel,
        }

        model_name = model_hparams['model']
        model = enabled_models[model_name]
        model = model(hp=model_hparams)

        return model

    def initialize_optimizer(self, model_hparams):
        enabled_optimizers = {'adamW': optim.AdamW,
                              'adam': optim.Adam,
                              'rAdam': optim.RAdam,
                              'sdg': optim.SGD}

        optimizer_name = model_hparams['optimizer']
        optimizer = enabled_optimizers[optimizer_name]
        weight_decay = model_hparams['weight_decay']
        learing_rate = model_hparams['learning_rate']

        optimizer = optimizer(params=self.parameters(), weight_decay=weight_decay, lr=learing_rate)

        return optimizer

    def forward(self, mols):
        return self.model(mols)

    def configure_optimizers(self):
        optimizer = self.initialize_optimizer(self.model_hparams)

        return optimizer

    def training_step(self, batch, batch_idx):
        preds = self.forward(batch)
        # preds = preds.squeeze(dim=-1)
        y = batch.y

        if self.model.is_multilabel():
            loss = self.multilabel_loss(self.loss_module, preds, y)
        else:
            loss = self.loss_module(preds, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        return {'loss': loss}

    # Runs after every epoch, the accuracy is used by the early stopping module as criteria
    def validation_step(self, batch, batch_idx): \
            # print('VALIDATION STEP')
        y = batch.y

        preds = self.forward(batch)
        # pred_probs = self.get_probs_as_tensor(batch)
        # pred_probs_list = pred_probs.tolist()

        if self.model.is_multilabel():
            loss = self.multilabel_loss(self.loss_module, preds, y)
        else:
            loss = self.loss_module(preds, y)

        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        metrics = {}
        #metrics = self.get_metrics(preds, y)
        metrics['val_loss'] = loss
        #self.log_multilabel_metrics(metrics)

    def get_metrics(self, preds, y):
        metrics = {}
        all_preds = []
        all_expected_labels = []

        precisions_accumulated = 0
        recalls_accumulated = 0
        f1s_accumulated = 0

        number_of_categories = len(preds)

        for category in preds:
            category_preds = preds[category].squeeze(dim=-1)
            category_preds = torch.sigmoid(category_preds)
            category_preds = [x.item() for x in category_preds]
            category_preds = [1 if x > 0.5 else 0 for x in category_preds]

            category_expected_labels = y[category]

            all_preds += category_preds
            all_expected_labels.append(category_expected_labels)

            category_precision = precision_score(category_expected_labels, category_preds)
            category_recall = recall_score(category_expected_labels, category_preds, zero_division=False)
            category_f1 = f1_score(category_expected_labels, category_preds, zero_division=False)

            precisions_accumulated += category_precision
            recalls_accumulated += category_recall
            f1s_accumulated += category_f1

            metrics[category] = {}
            metrics[category]['precision'] = category_precision
            metrics[category]['recall'] = category_recall
            metrics[category]['f1'] = category_f1

        #all_preds = torch.cat(all_preds, dim=0)
        all_expected_labels = torch.cat(all_expected_labels, dim=0)

        global_precision = precision_score(all_expected_labels, all_preds)
        global_recall = recall_score(all_expected_labels, all_preds, zero_division=False)
        global_f1 = f1_score(all_expected_labels, all_preds, zero_division=False)

        avg_precision = precisions_accumulated / number_of_categories
        avg_recall = recalls_accumulated / number_of_categories
        avg_f1 = f1s_accumulated / number_of_categories

        metrics['global'] = {}
        metrics['global']['precision'] = global_precision
        metrics['global']['recall'] = global_recall
        metrics['global']['f1'] = global_f1

        metrics['avg'] = {}
        metrics['avg']['precision'] = avg_precision
        metrics['avg']['recall'] = avg_recall
        metrics['avg']['f1'] = avg_f1

        return metrics

    def search_best_threshold(self, pred_probs_list, y):
        fpr, tpr, thresholds = roc_curve(y.tolist(), pred_probs_list)
        gmean = np.sqrt(tpr * (1 - fpr))
        index = np.argmax(gmean)

        best_thresh = round(thresholds[index], ndigits=4)
        gmeanOpt = round(gmean[index], ndigits=4)

        fprOpt = round(fpr[index], ndigits=4)
        tprOpt = round(tpr[index], ndigits=4)

        print('Best Threshold: {} with G-Mean: {}'.format(best_thresh, gmeanOpt))
        print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))

        return best_thresh

    # Runs only one time at the end of the training of the current model
    def test_step(self, batch, batch_idx):
        # print('VALIDATION STEP')
        y = batch.y

        preds = self.forward(batch)
        # pred_probs = self.get_probs_as_tensor(batch)
        # pred_probs_list = pred_probs.tolist()
        if self.model.is_multilabel():
            loss = self.multilabel_loss(self.loss_module, preds, y)
        else:
            loss = self.loss_module(preds, y)

        metrics = {}
        # metrics = self.get_metrics(preds, y)
        metrics['test_loss'] = loss.item()
        self.log_multilabel_metrics(metrics)

        # self.log('test_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        # self.log('test_avg_f1', metrics['global_f1'], prog_bar=True, logger=True, on_epoch=True, on_step=False)
        # self.log('test_avg_precision', metrics['global_precision'], prog_bar=True, logger=True, on_epoch=True,
        #        on_step=False)
        # self.log('test_avg_recall', metrics['global_recall'], prog_bar=True, logger=True, on_epoch=True, on_step=False)

    def log_multilabel_metrics(self, metrics):
        for category_metric_key in metrics:
            category_metric_values = metrics[category_metric_key]
            if 'loss' in category_metric_key:
                self.log(f'{category_metric_key}',
                         category_metric_values, prog_bar=True, logger=True, on_epoch=True,
                         on_step=False)
            else:
                for category_metric_values_name in category_metric_values:
                    self.log(f'{category_metric_key}_{category_metric_values_name}',
                             category_metric_values[category_metric_values_name], prog_bar=True, logger=True, on_epoch=True,
                             on_step=False)

    def get_probs_as_tensor(self, batch):
        preds = self.forward(batch)
        preds = preds.squeeze(dim=-1)
        pred_probs = torch.sigmoid(preds)

        return pred_probs

    def get_probs_as_list(self, batch):
        pred_probs = self.get_probs_as_tensor(batch)
        pred_probs = [x.item() for x in pred_probs]

        return pred_probs

    def log_metrics(self, prefix, y, pred_tags, loss, log_raw_values=False, best_thresh=None):

        total_positives = [x for x in y if x == 1]
        predicted_positives = [x for x in pred_tags if x == 1]

        correct_preds_pos = 0
        for pred, expected in zip(pred_tags, y):
            if expected == 1 and pred == expected:
                correct_preds_pos += 1

        precision = precision_score(y, pred_tags)
        recall = recall_score(y, pred_tags)
        f1_score_ = f1_score(y, pred_tags)
        acc = accuracy_score(y, pred_tags)

        if log_raw_values:
            self.log(f'total_pos', len(total_positives), prog_bar=True, logger=True, on_epoch=True, on_step=False)
            self.log(f'predicted_pos', len(predicted_positives), prog_bar=True, logger=True, on_epoch=True,
                     on_step=False)
            self.log(f'correct_pos', correct_preds_pos, prog_bar=True, logger=True, on_epoch=True,
                     on_step=False)

        self.log(f'{prefix}_acc', acc, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log(f'{prefix}_precision', precision, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log(f'{prefix}_recall', recall, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log(f'{prefix}_f1_score', f1_score_, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log(f'{prefix}_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        if prefix == 'test':
            # self.log('model', f'{self.model_hparams["model"]}', prog_bar=True, logger=True, on_epoch=True, on_step=False)
            self.log(f'best_thresh', best_thresh, prog_bar=True, logger=True, on_epoch=True, on_step=False)

    def multilabel_loss(self, loss_func, outputs, labels):
        loss = 0
        # class is the category which can take many values, for example fruity, herbal which can be 0 or 1
        for i, clazz in enumerate(outputs):
            # tomar el indice que maximiza el output y ese es nuestro label para cada clase
            if self.category_weights is not None:
                weight = torch.tensor(self.category_weights[clazz], dtype=torch.float)
                loss += torch.nn.NLLLoss(weight=weight, reduction='mean')(outputs[clazz], labels[clazz])
            else:
                outs = outputs[clazz].squeeze(dim=-1)
                target = labels[clazz]
                target = target.float()
                loss += loss_func(outs, target)

        return loss

    def save_activations(self, path):
        self.model.save_activations(path)
