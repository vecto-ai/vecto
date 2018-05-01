import tqdm


def is_in_jupyter():
    try:
        get_ipython
        return True
    except:
        return False


def get_tqdm(*args, **kwargs):
    if is_in_jupyter():
        return tqdm.tqdm_notebook(*args, **kwargs)
    return tqdm.tqdm(*args, **kwargs)
