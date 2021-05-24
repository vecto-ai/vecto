import logging
import os
import sys
import pandas
from pandas.io.json import json_normalize
from vecto.utils.data import load_json


logger = logging.getLogger(__name__)
logging.basicConfig(level="DEBUG")


# def clean_dic(data):
#     data_clean = {}
#     data_clean["task"] = data["experiment_setup"]["task"]
#     data_clean["embeddings"] = data["experiment_setup"]["embeddings"]["name"]
#     default_measurement = "accuracy"
#     if "default_measurement" in data["experiment_setup"]:
#         default_measurement = data["experiment_setup"]["default_measurement"]
#     else:
#         logger.warning(f"default_measurement not specified in ")
#     data_clean["result"] = data["result"][default_measurement]
#     return data_clean


# def df_from_file_bak(path):
#     logger.debug(f"processing {path}")
#     data = load_json(path)
#     data_clean = [clean_dic(x) for x in data]
#     # meta = [["experiment_setup", "task"],
#     #         ["experiment_setup", "subcategory"],
#     #         ["experiment_setup", "method"],
#     #         ["experiment_setup", "embeddings"]]
#     dframe = json_normalize(data_clean)
#     #if "details" in dframe:
#         #dframe.drop("details", axis="columns", inplace=True)
#     # default_measurement = "accuracy"
#     # try:
#     #     # TODO: check if default measurement is same for all experiments
#     #     default_measurement = dframe["experiment_setup.default_measurement"].unique()[0]
#     # except KeyError:
#     #     logger.warning(f"default_measurement not specified in {path}")
#     # dframe["result"] = dframe["result." + default_measurement]
#     # df["reciprocal_rank"] = 1 / (df["rank"] + 1)
#     return dframe

def df_from_file(path):
    data = load_json(path)
    # meta = [["experiment_setup", "task"],
    #         ["experiment_setup", "subcategory"],
    #         ["experiment_setup", "method"],
    #         ["experiment_setup", "embeddings"]]
    dframe = json_normalize(data)
    if "details" in dframe:
        dframe.drop("details", axis="columns", inplace=True)
    default_measurement = "accuracy"
    try:
        default_measurement = dframe["experiment_setup.default_measurement"].unique()[0]
    except KeyError:
        logger.warning(f"default_measurement not specified in {path}")
    dframe["result"] = dframe["result." + default_measurement]
    # df["reciprocal_rank"] = 1 / (df["rank"] + 1)
    return dframe


def df_from_dir(path):
    dfs = []
    for (dirpath, _, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith(".json"):
                full_path = os.path.join(dirpath, filename)
                try:
                    dfs.append(df_from_file(full_path))
                except KeyError:
                    logger.warning(f"error reading {full_path}")
    dframe = pandas.concat(dfs, sort=True)
    # print(dframe["experiment_setup.task"])
    return dframe


def get_filtered_dataframe(path, key_primary, key_secondary="experiment_setup.subcategory"):
    df = df_from_dir(path)
    print(df)
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
    print(unstacked)
    unstacked.plot.bar(rot=0)


if __name__ == "__main__":
    plot_accuracy(sys.argv[1],
                  key_primary="experiment_setup.task",
                  key_secondary="experiment_setup.embeddings.name")
    from matplotlib import pyplot as plt
    plt.savefig("results.pdf", bbox_inches="tight")
