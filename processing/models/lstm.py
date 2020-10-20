from .pytorch_tools.gradient_reversal import RevGrad
import torch
import os
from processing.models.deep_tl_predictor import DeepTLPredictor
import numpy as np
import torch.nn as nn
from .pytorch_tools.training import fit, predict
from captum.attr import IntegratedGradients


class LSTM(DeepTLPredictor):
    def __init__(self, subject, ph, params, train, valid, test):
        super().__init__(subject, ph, params, train, valid, test)

        self.model = self.LSTM_Module(self.input_shape, self.params["hidden"], self.params["dropout_weights"],
                                      self.params["dropout_layer"], self.params["domain_adversarial"], self.n_domains)
        self.model.cuda()
        self.loss_func = self._compute_loss_func()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"], weight_decay=self.params["l2"])

    def fit(self):
        x_train, y_train, t_train = self._str2dataset("train")
        x_valid, y_valid, t_valid = self._str2dataset("valid")
        train_ds = self._to_tensor_ds(x_train, y_train)
        valid_ds = self._to_tensor_ds(x_valid, y_valid)

        fit(self.params["epochs"], self.params["batch_size"], self.model, self.loss_func, self.opt, train_ds, valid_ds,
            self.params["patience"], self.checkpoint_file)

    def predict(self, dataset, clear=False):
        # get the data for which we make the predictions
        x, y, t = self._str2dataset(dataset)
        ds = self._to_tensor_ds(x, y)

        # create the model
        self.model.load_state_dict(torch.load(self.checkpoint_file))

        if self.params["domain_adversarial"]:
            [y_trues_glucose, y_trues_subject], [y_preds_glucose, y_preds_subject] = predict(self.model, ds)
            results = self._format_results_source(y_trues_glucose, y_trues_subject, y_preds_glucose, y_preds_subject, t)
        else:
            y_true, y_pred = predict(self.model, ds)
            results = self._format_results(y_true, y_pred, t)

        if clear:
            self._clear_checkpoint()

        return results

    def save(self, save_file, clear=True):
        no_da_lstm = self.LSTM_Module(self.input_shape, self.params["hidden"], self.params["dropout_weights"],
                                      self.params["dropout_layer"], False, 0)
        self.model.load_state_dict(torch.load(self.checkpoint_file))
        no_da_lstm.encoder.load_state_dict(self.model.encoder.state_dict())
        no_da_lstm.regressor.load_state_dict(self.model.regressor.state_dict())
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        torch.save(no_da_lstm.state_dict(), save_file)
        if clear:
            self._clear_checkpoint()

    def _compute_input_shape(self):
        x_train, _, _ = self._str2dataset("train")
        return x_train.shape[2]

    def extract_features(self, dataset, file):
        x, y, _ = self._str2dataset(dataset)
        self.model.load_state_dict(torch.load(file))
        self.model.eval()
        features = self.model.encoder(torch.Tensor(x).cuda())[0].detach().cpu().numpy()
        features = features.reshape(features.shape[0], -1)

        return [features, y]

    def integrated_gradients(self, dataset, file):
        self.model.load_state_dict(torch.load(file))
        x, y, _ = self._str2dataset(dataset)
        baselines = np.zeros(x.shape)
        names = self.input_names
        for i in range(len(names)):
            if names[i] == "glucose":
                mean = np.mean(y)
                baselines[:, :, i] = mean * np.ones(x.shape[0:2])
            # if names[i] == "mets":
            #     baselines[:, :, i] = mean * np.ones(x.shape[0:2])

        x = torch.Tensor(x).cuda()
        x.requires_grad_()
        baselines = torch.Tensor(baselines).cuda()
        self.model.train()
        ig = IntegratedGradients(self.model)
        attr = []
        for i in range(len(x) // 10):
            attr_i = ig.attribute(x[i * 10:(i + 1) * 10], baselines=baselines[i * 10:(i + 1) * 10], n_steps=50)
            attr.append(attr_i.cpu().detach().numpy())
        attr_i = ig.attribute(x[(i + 1) * 10:len(x)], baselines=baselines[(i + 1) * 10:len(x)], n_steps=50)
        attr.append(attr_i.cpu().detach().numpy())
        attr = np.concatenate(attr, axis=0)
        return attr

    class LSTM_Module(nn.Module):

        def __init__(self, n_in, neurons, dropout_weights, dropout_layer, domain_adversarial=False, n_domains=1):
            super().__init__()

            self.encoder = nn.LSTM(n_in, neurons[0], len(neurons), dropout=dropout_layer, batch_first=True)
            self.regressor = nn.Linear(neurons[-1], 1)
            self.domain_classifier = nn.Sequential(
                RevGrad(),
                nn.Linear(neurons[-1], n_domains),
                nn.LogSoftmax(dim=1),
            ) if domain_adversarial else None

        def forward(self, xb):
            features, _ = self.encoder(xb)
            prediction = self.regressor(features[:, -1])
            if self.domain_classifier is not None:
                # domain = self.domain_classifier(self.GradReverse.apply(features[:,-1]))
                domain = self.domain_classifier(features[:, -1])
                return prediction.reshape((-1)), domain
            else:
                return prediction.reshape((-1))
