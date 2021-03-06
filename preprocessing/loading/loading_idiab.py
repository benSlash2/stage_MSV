import pandas as pd
from os.path import join
import misc.constants as cs


def load_idiab(dataset, subject):
    """
    Load a IDIAB file into a dataframe
    :param dataset: name of dataset
    :param subject: name of subject
    :return: dataframe
    """
    df = pd.read_csv(join(cs.path, "data", dataset, "IDIAB_steps__" + subject + ".csv"), header=0)
    df = df.drop("index", axis=1)
    df.datetime = pd.to_datetime(df.datetime)
    return df
