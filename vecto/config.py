from traitlets.config.loader import load_pyconfig_files
import os.path


def load_config():
    default_dir = os.path.expanduser("~/.vecto/")
    if os.path.isfile(os.path.join(default_dir, 'config.py')):
        c = load_pyconfig_files(['config.py'], default_dir)
        return c
    else:
        raise ValueError('configuration file not find, please create one in ~/.vecto/config.py')
        exit(0)
