from .loading_idiab import load_idiab
from .loading_ohio import load_ohio
from .loading_t1dms import load_t1dms


def load(dataset, subject, day_len=None):
    if "idiab" in dataset:
        return load_idiab(dataset, subject)
    elif "ohio" in dataset:
        return load_ohio(dataset, subject)
    else:
        return load_t1dms(dataset, subject, day_len)
