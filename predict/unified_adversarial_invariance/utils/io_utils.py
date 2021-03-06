import pickle
import shutil
import os


def pickle_write(obj, path, protocol=pickle.HIGHEST_PROTOCOL):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)


def pickle_read(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def localize_file(remote_path):
    local_dir = os.environ['TMPDIR']
    filename = os.path.basename(remote_path)
    local_path = os.path.join(local_dir, filename)
    if not os.path.exists(local_path):
        shutil.copy(remote_path, local_path)
    return local_path
