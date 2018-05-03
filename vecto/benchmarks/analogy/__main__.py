import sys
import yaml
from .analogy import Analogy


def main():
    if len(sys.argv) > 1:
        path_config = sys.argv[1]
    else:
        print("usage: python3 -m vecto.benchmarks.analogy <config file>")
        print("config file example can be found at ")
        print("https://github.com/undertherain/vsmlib/blob/master/vsmlib/benchmarks/analogy/config_analogy.yaml")
        return


    with open(path_config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    options = {}
    options["name_method"] = cfg["method"]
    options["exclude"] = cfg["exclude"]
    options["path_dataset"] = cfg["path_dataset"]
    options["path_results"] = cfg["path_results"]
    options["normalize"] = cfg["normalize"]
    options["path_vectors"] = cfg["path_vectors"]

    analogy = Analogy()
    analogy.run()


if __name__ == "__main__":
    main()
