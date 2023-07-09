import os

from prettytable import PrettyTable

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import ProgressBar

from modules.lightning_module import GNNLightning


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def train_graph_classifier(hyperparams, root_dir, train_loader, test_loader, validation_loader):
    model = GNNLightning(model_hparams=hyperparams)

    os.makedirs(root_dir, exist_ok=True)

    progress_bar = ProgressBar()
    progress_bar.test_progress_bar = True

    early_stopping = EarlyStopping(monitor='train_loss',
                                   min_delta=hyperparams['early_stopping_min_delta'],
                                   patience=hyperparams['early_stopping_patience'],
                                   verbose=True,
                                   check_on_train_epoch_end=True)

    # logger = TensorBoardLogger(name='tensorboard_logs', save_dir=root_dir)
    trainer = Trainer(default_root_dir=root_dir,
                      callbacks=[ModelCheckpoint(save_weights_only=False,
                                                 mode="max",
                                                 monitor=hyperparams['checkpoint_metric'],
                                                 save_top_k=1),
                                 early_stopping],
                      max_epochs=hyperparams['max_epochs'],
                      log_every_n_steps=1,
                      check_val_every_n_epoch=1)

    count_parameters(model)
    trainer.fit(model, train_loader, validation_loader)
    best_thresh = model.get_threshold()

    best_model_path = trainer.checkpoint_callback.best_model_path
    print(('BEST MODEL PATH: ' + str(best_model_path)))

    # Test best model on validation and test set
    # train_result = trainer.test(model, test_dataloaders=train_loader, verbose=False)

    best_model = GNNLightning.load_from_checkpoint(best_model_path)
    best_model.best_threshold = best_thresh
    model_name = best_model.model_name

    result = trainer.test(best_model, dataloaders=test_loader, verbose=True)
    result[0]['model'] = model_name

    return best_model, result
