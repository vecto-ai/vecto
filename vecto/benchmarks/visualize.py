import os
import pandas
from pandas.io.json import json_normalize
from vecto.utils.data import load_json


def df_from_file(path):
    data = load_json(path)
    meta = [["experiment_setup", "subcategory"], ["experiment_setup", "method"], ["experiment_setup", "embeddings"]]
    dframe = json_normalize(data, meta=meta)
    if "details" in dframe:
        dframe.drop("details", axis="columns", inplace=True)
    dframe["result"] = dframe["result." + dframe["experiment_setup.default_measurement"].unique()[0]]
    # df["reciprocal_rank"] = 1 / (df["rank"] + 1)
    return dframe


def df_from_dir(path):
    dfs = []
    for (dirpath, _, filenames) in os.walk(path):
        for filename in filenames:
            dfs.append(df_from_file(os.path.join(dirpath, filename)))
    dframe = pandas.concat(dfs)
    return dframe


def get_filtered_dataframe(path, key_primary, key_secondary="experiment_setup.subcategory"):
    df = df_from_dir(path)
    groupby_items = [key_secondary, key_primary]

    group = df.groupby(groupby_items)
    means = group.mean()
    means.reset_index(inplace=True)
    means = means.loc[:, groupby_items + ["result"]]
    # means = pandas.concat((means, means))
    unstacked = means.groupby(groupby_items)['result'].aggregate('first').unstack()
    return unstacked


def plot_accuracy(path, key_primary="experiment_setup.method",
                  key_secondary="experiment_setup.subcategory"):
    unstacked = get_filtered_dataframe(path, key_primary, key_secondary)
    unstacked.plot.bar(rot=0)
