import misc.constants as cs
from processing.models.predictor import Predictor
from sklearn.tree import DecisionTreeRegressor
import misc.constants as cs
import os
from joblib import dump, load

class DT(Predictor):
    def fit(self):
        # get training data
        x, y, t = self._str2dataset("train")

        # define the model
        self.model = DecisionTreeRegressor(
            criterion="mse",
            max_depth=self.params["max_depth"],
            min_samples_split=self.params["min_samples_split"],
            random_state=cs.seed
        )

        # fit the model
        self.model.fit(x, y)

    def predict(self, dataset):
        # get the data for which we make the predictions
        x, y, t = self._str2dataset(dataset)

        # predict
        y_pred = self.model.predict(x)
        y_true = y.values

        return self._format_results(y_true, y_pred, t)


    def save(self, file):
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        dump(self.model,filename=file)

    def load(self, file):
        self.model = load(filename=file)