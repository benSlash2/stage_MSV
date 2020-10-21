import torch
import os
from processing.models.deep_predictor import DeepPredictor
import torch.nn as nn
from processing.models.pytorch_tools.training import fit, predict


class RETAIN(DeepPredictor):
    def __init__(self, subject, ph, params, train, valid, test):
        super().__init__(subject, ph, params, train, valid, test)

        self.model = self.RetainModule(self.input_shape, self.params["n_features_emb"], self.params["n_hidden_rnn"],
                                       self.params["n_layers_rnn"], self.params["emb_dropout"],
                                       self.params["ctx_dropout"], self.params["reverse_time"],
                                       self.params["bidirectional"])

        self.model_parameters = [
            {'params': self.model.embeddings.parameters()},
            {'params': self.model.rnn_alpha.parameters(), 'weight_decay': 0},
            {'params': self.model.rnn_beta.parameters(), 'weight_decay': 0},
            {'params': self.model.alpha.parameters()},
            {'params': self.model.beta.parameters()},
            {'params': self.model.output.parameters()},
        ]

        self.model.cuda()
        self.loss_func = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model_parameters, lr=self.params["lr"], weight_decay=self.params["l2"])

    def fit(self):
        x_train, y_train, t_train = self._str2dataset("train")
        x_valid, y_valid, t_valid = self._str2dataset("valid")
        train_ds = self._to_tensor_ds(x_train, y_train)
        valid_ds = self._to_tensor_ds(x_valid, y_valid)

        fit(self.params["epochs"], self.params["batch_size"], self.model, self.loss_func, self.opt, train_ds, valid_ds,
            self.params["patience"], self.checkpoint_file)

    def predict(self, dataset, clear=True):
        # get the data for which we make the predictions
        x, y, t = self._str2dataset(dataset)
        ds = self._to_tensor_ds(x, y)

        # create the model
        self.model.load_state_dict(torch.load(self.checkpoint_file))

        y_true, y_pred = predict(self.model, ds)
        results = self._format_results(y_true, y_pred, t)

        if clear:
            self._clear_checkpoint()

        return results

    def save(self, save_file):
        self.model.load_state_dict(torch.load(self.checkpoint_file))
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        torch.save(self.model.state_dict(), save_file)

    def _compute_input_shape(self):
        x_train, _, _ = self._str2dataset("train")
        return x_train.shape[2]

    def _reshape(self, data):
        x, y, t = super()._reshape(data)
        return x, y, t

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
                     reverse_time=True, bidirectional=False):
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
            )

            self.beta = nn.Sequential(
                nn.Linear(n_hidden_rnn * (1 + int(bidirectional)), n_features_emb, bias=True),
                nn.Tanh()
            )

            self.output = nn.Linear(n_features_emb, 1, bias=True)

        def forward(self, xb):
            emb = self.compute_embeddings(xb)
            alpha, beta = self.compute_alpha_beta(emb)
            c = self.compute_c(emb, alpha, beta)

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
