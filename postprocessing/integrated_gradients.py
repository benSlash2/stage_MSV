import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import numpy as np
import misc
import misc.constants as cs
import pickle
from pathlib import Path
from preprocessing.preprocessing import preprocessing
from processing.models.retain_atl import RETAIN_ATL
from misc.utils import locate_params
import pandas as pd
from captum.attr import visualization as viz


def print_ig_sample(features, attr, results, sample=0):
    image = features[sample]
    attr_sample = attr[sample]
    true, pred = results.iloc[sample].values

    default_cmap = LinearSegmentedColormap.from_list('custom blue', [(0, '#ffffff'), (0.25, '#000000'),
                                                                     (1, '#000000')], N=256)

    # _ = viz.visualize_image_attr(attr_sample, image, method='heat_map', cmap=default_cmap, show_colorbar=True,
    #                              sign='positive', outlier_perc=1)

    _ = viz.visualize_image_attr_multiple(attr, features, ["original_image", "heat_map"], ["all", "absolute_value"],
                                          cmap=default_cmap, show_colorbar=True, titles=["true cd =" + str(true),
                                                                                         "predicted cd = " + str(pred)])
