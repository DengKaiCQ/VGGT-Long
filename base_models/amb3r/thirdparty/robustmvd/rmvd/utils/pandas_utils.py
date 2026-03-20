# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/utils/pandas_utils.py

import pandas as pd


def prepend_level(df, level_name, level, axis=0):
    return pd.concat({level: df}, axis=axis, names=[level_name])
