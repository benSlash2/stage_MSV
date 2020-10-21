import numpy as np
from processing.models.pytorch_tools.gradient_reversal import RevGrad
import torch
import os
from processing.models.deep_tl_predictor import DeepTlPredictor
import torch.nn as nn
from processing.models.pytorch_tools.training import fit, predict


class RetainATL(DeepTlPredictor):
    def __init__(self, subject, ph, params, train, valid, test):
        super().__init__(subject, ph, params, train, valid, test)

        self.model = self.RetainModule(self.input_shape, self.params["n_features_emb"], self.params["n_hidden_rnn"],
                                       self.params["n_layers_rnn"], self.params["emb_dropout"],
                                       self.params["ctx_dropout"], self.params["reverse_time"],
                                       self.params["bidirectional"], self.n_domains)

        self.model_parameters = [
            {'params': self.model.embeddings.parameters()},
            {'params': self.model.rnn_alpha.parameters(), 'weight_decay': 0},
            {'params': self.model.rnn_beta.parameters(), 'weight_decay': 0},
            {'params': self.model.alpha.parameters()},
            {'params': self.model.beta.parameters()},
            {'params': self.model.output.parameters()},
        ]

        self.model.cuda()
        self.loss_func = self._compute_loss_func()
        self.opt = torch.optim.Adam(self.model_parameters, lr=self.params["lr"], weight_decay=self.params["l2"])

    def fit(self):
        x_train, y_train, t_train = self._str2dataset("train")
        x_valid, y_valid, t_valid = self._str2dataset("valid")
        train_ds = self._to_tensor_ds(x_train, y_train)
        valid_ds = self._to_tensor_ds(x_valid, y_valid)

        fit(self.params["epochs"], self.params["batch_size"], self.model, self.loss_func, self.opt, train_ds, valid_ds,
            self.params["patience"], self.checkpoint_file)

    def loss_init(self):
        x_train, y_train, t_train = self._str2dataset("train")
        x_valid, y_valid, t_valid = self._str2dataset("valid")
        train_ds = self._to_tensor_ds(x_train, y_train)
        valid_ds = self._to_tensor_ds(x_valid, y_valid)

        self.loss_func = self._compute_loss_func()

        self.opt = torch.optim.Adam(self.model_parameters, lr=self.params["lr"], weight_decay=self.params["l2"])

        loss_init(self.params["epochs"], self.params["batch_size"], self.model, self.loss_func, self.opt, train_ds,
                  valid_ds, self.params["patience"], self.checkpoint_file)

    def predict(self, dataset, clear=True):
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

    def save(self, file):
        self.model.load_state_dict(torch.load(self.checkpoint_file))
        no_da_retain = self.RetainModule(self.input_shape, self.params["n_features_emb"], self.params["n_hidden_rnn"],
                                         self.params["n_layers_rnn"], self.params["emb_dropout"],
                                         self.params["ctx_dropout"], self.params["reverse_time"],
                                         self.params["bidirectional"], 0)
        no_da_retain.embeddings.load_state_dict(self.model.embeddings.state_dict())
        no_da_retain.rnn_alpha.load_state_dict(self.model.rnn_alpha.state_dict())
        no_da_retain.rnn_beta.load_state_dict(self.model.rnn_beta.state_dict())
        no_da_retain.alpha.load_state_dict(self.model.alpha.state_dict())
        no_da_retain.beta.load_state_dict(self.model.beta.state_dict())
        no_da_retain.output.load_state_dict(self.model.output.state_dict())
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        torch.save(no_da_retain.state_dict(), file)

    def _compute_input_shape(self):
        x_train, _, _ = self._str2dataset("train")
        return x_train.shape[2]

    def _reshape(self, data):
        x, y, t = super()._reshape(data)
        return x, y, t

    def get_attention_weights(self, dataset):
        x, y, t = self._str2dataset(dataset)
        ds = self._to_tensor_ds(x, y)

        emb = self.model.compute_embeddings(ds[0])
        alpha, beta = self.model.compute_alpha_beta(emb)

        return alpha, beta

    def get_embeddings_attr(self):
        return self.model.embeddings.weight

    def get_output_attr(self):
        return self.model.output.weight, self.model.output.bias

    def contribution(self, dataset):
        x, y, t = self._str2dataset(dataset)
        # ds = self._to_tensor_ds(x, y)

        xb = torch.Tensor(x).cuda()
        # xb = torch.Tensor(x)

        emb = self.model.compute_embeddings(xb)
        alpha, beta = self.model.compute_alpha_beta(emb)

        w_emb = self.model.embeddings.weight
        w_out, b_out = self.model.output.weight, self.model.output.bias

        contrib = []
        for i in range(xb.shape[1]):
            contrib_i = []
            for j in range(xb.shape[2]):
                contrib_ij = (alpha[:, i, :] * (
                    torch.matmul((beta[:, i, :] * w_emb[:, j]), w_out.transpose(1, 0)))).squeeze() * xb[:, i, j]
                contrib_i.append(contrib_ij.detach().cpu().numpy())
            contrib.append(contrib_i)
        contrib = np.array(contrib)

        return contrib.transpose(2, 0, 1)

    def contribution_an(self, dataset):
        contrib = self.contribution(dataset)
        absolute_contrib = np.abs(contrib)
        sum_contrib = np.expand_dims(absolute_contrib.sum(1).sum(1), axis=[1, 2])
        contrib_an = absolute_contrib / sum_contrib
        return contrib_an

    def extract_features(self, dataset, file):
        x, y, _ = self._str2dataset(dataset)
        self.model.load_state_dict(torch.load(file))
        self.model.eval()

        xb = torch.Tensor(x).cuda()
        emb = self.model.compute_embeddings(xb)
        alpha, beta = self.model.compute_alpha_beta(emb)
        c = self.model.compute_c(emb, alpha, beta)

        c = c.detach().cpu().numpy().reshape(c.shape[0], -1)

        return [c, y]

    class RetainModule(nn.Module):

        def __init__(self, n_in, n_features_emb, n_hidden_rnn, n_layers_rnn, emb_dropout, ctx_dropout,
                     reverse_time=True, bidirectional=False, adversarial_domains=0):
            super().__init__()

            self.embeddings = nn.Linear(n_in, n_features_emb, bias=False)
            self.rnn_alpha = nn.LSTM(n_features_emb, n_hidden_rnn, n_layers_rnn, batch_first=True,
                                     bidirectional=bidirectional)
            self.rnn_beta = nn.LSTM(n_features_emb, n_hidden_rnn, n_layers_rnn, batch_first=True,
                                    bidirectional=bidirectional)
            self.reverse_time = reverse_time
            self.emb_dropout = nn.Dropout(emb_dropout)
            self.ctx_dropout = nn.Dropout(ctx_dropout)

            self.alpha = nn.Sequential(
                nn.Linear(n_hidden_rnn * (1 + int(bidirectional)), 1, bias=True),
                nn.Softmax(dim=1)
                # Sparsemax(dim=1)
            )

            self.beta = nn.Sequential(
                nn.Linear(n_hidden_rnn * (1 + int(bidirectional)), n_features_emb, bias=True),
                nn.Tanh()
            )

            self.output = nn.Linear(n_features_emb, 1, bias=True)

            if adversarial_domains != 0:
                self.domain_classifier = nn.Sequential(
                    RevGrad(),
                    nn.Linear(n_features_emb, adversarial_domains, bias=True),
                    nn.LogSoftmax(dim=1)
                )
            else:
                self.domain_classifier = None

        def forward(self, xb):
            emb = self.compute_embeddings(xb)
            alpha, beta = self.compute_alpha_beta(emb)
            c = self.compute_c(emb, alpha, beta)

            if self.domain_classifier is not None:
                return self.output(c).reshape((-1)), self.domain_classifier(c)
            else:
                return self.output(c).reshape((-1))

        def compute_embeddings(self, xb):
            emb = self.embeddings(xb)
            if self.reverse_time:
                emb = emb.flip(dims=[1])
            return self.emb_dropout(emb)

        def compute_alpha_beta(self, emb):
            return self.alpha(self.rnn_alpha(emb)[0]), self.beta(self.rnn_beta(emb)[0])

        def compute_c(self, emb, alpha, beta):
            return self.ctx_dropout((alpha * beta * emb).sum(dim=1))
