import numpy as np
import torch
from misc.utils import printd
from misc.constants import path
import os
import copy

""" Credit : https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py """

class EarlyStopping:
    """Early stops the training_old if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, save_n_epochs=10, path=os.path.join(path,"checkpoints.pt"), verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
        self.save_n_epochs = save_n_epochs

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.val_loss_min = val_loss
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            printd(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.val_loss_min = val_loss
            self.counter = 0

        # if epoch % self.save_n_epochs == 0:
        #     self.save()

    # def save_checkpoint(self, val_loss, model):
    #     '''Saves model when validation loss decrease.'''
    #     if self.verbose:
    #         printd(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    #     torch.save(model.state_dict(), self.path)
    #     self.val_loss_min = val_loss

    def save(self):
        torch.save(self.best_model, self.path)