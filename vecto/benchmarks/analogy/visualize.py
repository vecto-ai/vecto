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
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            dfs.append(df_from_file(os.path.join(dirpath, filename)))
    dframe = pandas.concat(dfs)
    return dframe


def plot_accuracy(path, key_primary="experiment_setup.method",
                  key_secondary="experiment_setup.subcategory"):
    df = df_from_dir(path)
    group = df.groupby([key_primary, key_secondary])
    means = group.mean()
    print(means)
    means.reset_index(inplace=True)
    means = means.loc[:, [key_primary, key_secondary, "result"]]
    unstacked = means.groupby([key_secondary, key_primary])['result'].aggregate('first').unstack()
    unstacked.plot.bar(rot=0)


def main():
    plot_accuracy()


if __name__ == "__main__":
    main()
