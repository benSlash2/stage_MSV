from torch.utils.data import TensorDataset
import torch
from misc.utils import printd
import os
from processing.models.predictor import Predictor
import numpy as np
import misc.constants as cs
import torch.nn as nn
from .pytorch_tools.training import fit, predict


class LSTM_ICPRAI(Predictor):
    def fit(self):
        # get training data
        x_train, y_train, t_train = self._str2dataset("train")
        x_valid, y_valid, t_valid = self._str2dataset("valid")

        # save model
        rnd = np.random.randint(1e7)
        self.checkpoint_file = os.path.join(cs.path, "tmp", "checkpoints", "lstm_" + str(rnd) + ".pt")
        printd("Saved model's file:", self.checkpoint_file)

        self.model = self.LSTM_Module(x_train.shape[2], self.params["hidden"],
                                      self.params["dropout_weights"], self.params["dropout_layer"])
        self.model.cuda()
        self.loss_func = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"], weight_decay=self.params["l2"])

        train_ds = self.to_dataset(x_train, y_train)
        valid_ds = self.to_dataset(x_valid, y_valid)

        fit(self.params["epochs"], self.params["batch_size"], self.model, self.loss_func, self.opt, train_ds, valid_ds,
            self.params["patience"], self.checkpoint_file)

    def predict(self, dataset):
        # get the data for which we make the predictions
        x, y, t = self._str2dataset(dataset)
        ds = self.to_dataset(x, y)

        # create the model
        self.model.load_state_dict(torch.load(self.checkpoint_file))

        # predict
        y_true, y_pred = predict(self.model, ds)

        return self._format_results(y_true, y_pred, t)

    def _reshape(self, data):
        # extract data from data df
        t = data["datetime"]
        y = data["y"]

        g = data.loc[:, [col for col in data.columns if "glucose" in col]].values
        cho = data.loc[:, [col for col in data.columns if "CHO" in col]].values
        ins = data.loc[:, [col for col in data.columns if "insulin" in col]].values

        # reshape timeseties in (n_samples, hist, 1) shape and concatenate
        g = g.reshape(-1, g.shape[1], 1)
        cho = cho.reshape(-1, g.shape[1], 1)
        ins = ins.reshape(-1, g.shape[1], 1)

        x = np.concatenate([g, cho, ins], axis=2)

        return x, y, t

    class LSTM_Module(nn.Module):
        def __init__(self, n_in, neurons, dropout_weights, dropout_layer):
            super().__init__()

            self.embeddings = nn.Linear(n_in, 64, bias=False)

            self.lstm = nn.LSTM(64, neurons[0], len(neurons), dropout=dropout_layer, batch_first=True)

            self.linear = nn.Linear(neurons[-1], 1)

        def forward(self, xb):
            xb = self.embeddings(xb)
            if self.lstm.__class__.__name__ == "LSTM":
                xb, _ = self.lstm(xb)
            else:
                for lstm_, dropout_ in zip(self.lstm, self.dropouts):
                    xb = lstm_(xb)[0]
            xb = self.linear(xb[:, -1, :])

            return xb.reshape(-1)

    def to_dataset(self, x, y):
        return TensorDataset(torch.Tensor(x).cuda(), torch.Tensor(y).cuda())
