"""
JSON
====

Defines functions to load and save json files.

The code for compression is from
https://github.com/LucaCappelletti94/compress_json.
"""

import json
import inspect
import importlib
import cloudpickle
import pickle
import codecs
import gzip
import lzma
import bz2
import os

import numpy as np
import pandas as pd
from pandas.compat.pickle_compat import loads as pandasloads

from dreye.constants import ureg
from dreye.err import DreyeError
from dreye.io import ignore


ARRAY_PREFIX = "__ARRAY__"
RECARRAY_PREFIX = "__RECARRAY__"
DTYPE_PREFIX = "__DTYPE__"
SERIES_PREFIX = "__SERIES__"
DFRAME_PREFIX = "__FRAME__"
DFRAME_PREFIX_PLUS = "__FRAMEPLUS__"
PINT_PREFIX = "__PINT__"
ANY_PREFIX = "__ANY__"
QUANT_PREFIX = "__QUANT__"
DREYE_PREFIX = "__DREYE__"
DICTABLE_PREFIX = "__DICTABLE__"
MODULE_PREFIX = "__MODULE__"
CLASS_PREFIX = "__CLASS__"
DATA_PREFIX = "__DATA__"
PICKLED_PREFIX = "__PICKLE__"
FILEPATH_PREFIX = "__FILE__"

_DEFAULT_EXTENSION_MAP = {
    "json": "json",
    "gz": "gzip",
    "gzip": "gzip",
    "bz": "bz2",
    "bz2": "bz2",
    "lzma": "lzma",
    "pkl": "pickle"
}

_DEFAULT_COMPRESSION_WRITE_MODES = {
    "json": "w",
    "gzip": "wt",
    "bz2": "wt",
    "lzma": "wt"
}

_DEFAULT_COMPRESSION_READ_MODES = {
    "json": "r",
    "gzip": "rt",
    "bz2": "rt",
    "lzma": "rt"
}


def get_compression_write_mode(compression: str) -> str:
    """Return mode for opening file buffer for writing."""
    return _DEFAULT_COMPRESSION_WRITE_MODES.get(compression, "w")


def get_compression_read_mode(compression: str) -> str:
    """Return mode for opening file buffer for reading."""
    return _DEFAULT_COMPRESSION_READ_MODES.get(compression, "r")


def infer_compression_from_filename(filename: str) -> str:
    """Return the compression protocal inferred from given filename.
    Parameters
    ----------
    filename: str
        The filename for which to infer the compression protocol
    """
    return _DEFAULT_EXTENSION_MAP.get(filename.split(".")[-1], None)


def read_json(filename):
    """
    Read a JSON file.

    Parameters
    ----------
    filename : str
        location of file.

    Returns
    -------
    data : dict
        Data in JSON file.
    """

    compression = infer_compression_from_filename(filename)
    mode = get_compression_read_mode(compression)
    if compression is None or compression == "json":
        file = open(filename, mode=mode, encoding="utf-8")
    elif compression == "gzip":
        file = gzip.open(filename, mode=mode, encoding="utf-8")
    elif compression == "bz2":
        file = bz2.open(filename, mode=mode, encoding="utf-8")
    elif compression == "lzma":
        file = lzma.open(filename, mode=mode, encoding="utf-8")

    with file as f:
        data = json.load(f, object_hook=deserializer)
    return data


def load_json(obj):
    """
    Read JSON string.
    """

    return json.loads(obj, object_hook=deserializer)


def write_json(filename, data):
    """
    Write to a JSON file.

    Parameters
    ----------
    filename : str
        location of file.
    data : dict
        Data to write to JSON file.
    """

    compression = infer_compression_from_filename(filename)
    mode = get_compression_write_mode(compression)
    if compression is None or compression == "json":
        file = open(filename, mode=mode, encoding="utf-8")
    elif compression == "gzip":
        file = gzip.open(filename, mode=mode, encoding="utf-8")
    elif compression == "bz2":
        file = bz2.open(filename, mode=mode, encoding="utf-8")
    elif compression == "lzma":
        file = lzma.open(filename, mode=mode, encoding="utf-8")

    with file as f:
        json.dump(data, f, indent=4, default=serializer)


def dump_json(obj):
    """
    Return JSON string.
    """

    return json.dumps(obj, indent=4, default=serializer)


def _dataframe_serializer(obj, is_series=False):
    data = obj.to_dict('split')
    data['index_names'] = list(obj.index.names)
    data['columns_names'] = list(obj.columns.names)
    data['index_multi'] = isinstance(obj.index, pd.MultiIndex)
    data['columns_multi'] = isinstance(obj.columns, pd.MultiIndex)
    data['is_series'] = is_series
    return {
        DFRAME_PREFIX_PLUS: data
    }


def _class_serializer(obj):
    cls = type(obj)
    module_name = inspect.getmodule(cls).__name__
    return cls, module_name, {
        MODULE_PREFIX: module_name,
        CLASS_PREFIX: cls.__name__
    }


def serializer(obj):
    """
    Serializer of numpy, pandas objects and dreye objects etc.
    """
    # TODO NaN handling in numpy arrays

    if isinstance(obj, np.ndarray) and obj.dtype.fields is not None:
        # ATTENTION: https://stackoverflow.com/questions/24814595/
        #            is-there-any-way-to-determine-if-an
        #            -numpy-array-is-record-structure-array
        obj = {
            RECARRAY_PREFIX: _dataframe_serializer(pd.DataFrame(obj), False)
        }
    elif isinstance(obj, np.ndarray):
        obj = {ARRAY_PREFIX: obj.tolist()}
    elif isinstance(obj, pd.Series):
        obj = _dataframe_serializer(obj.to_frame(), True)
    elif isinstance(obj, pd.DataFrame):
        obj = _dataframe_serializer(obj, False)
    elif isinstance(obj, np.dtype):
        obj = {DTYPE_PREFIX: str(obj)}
    elif isinstance(obj, ureg.Unit):
        obj = {PINT_PREFIX: str(obj)}
    elif isinstance(obj, ureg.Quantity):
        obj = {QUANT_PREFIX: [obj.magnitude, obj.units]}  # TODO convert NaNs to None
    elif hasattr(obj, 'to_dict') and hasattr(obj, 'from_dict'):
        cls, module_name, dictionary = _class_serializer(obj)
        if module_name.split('.')[0] == 'dreye':
            obj = {
                DREYE_PREFIX: {
                    DATA_PREFIX: obj.to_dict(),
                    **dictionary
                }
            }
        else:
            # possibly custom subclasses
            obj = {
                DICTABLE_PREFIX: {
                    PICKLED_PREFIX: codecs.encode(
                        spickledumps(cls), "base64"
                    ).decode(),
                    DATA_PREFIX: obj.to_dict(),
                    **dictionary
                }
            }
    # TODO saving functions separately?
    else:
        # TODO check any and throw warnings
        obj = {
            PICKLED_PREFIX: codecs.encode(
                spickledumps(obj), "base64"
            ).decode()
        }

    return obj


def _class_deserializer(ele):
    try:
        return getattr(
            importlib.import_module(ele[MODULE_PREFIX]),
            ele[CLASS_PREFIX]
        )
    except Exception as e:
        from dreye.io.deprecated_class_conversions import _deprecated_classes
        if ele[CLASS_PREFIX] in _deprecated_classes:
            return _deprecated_classes[ele[CLASS_PREFIX]]
        else:
            raise e


def _deserialize_index(data, name='index'):
    if data[f'{name}_multi']:
        index = pd.MultiIndex.from_tuples(
            data[name], names=data[f'{name}_names']
        )
    else:
        index = pd.Index(
            data[name], name=data[f'{name}_names'][0],
            tupleize_cols=False
        )
    return index


def _dataframe_deserializer(data):
    index = _deserialize_index(data, name='index')
    columns = _deserialize_index(data, name='columns')
    df = pd.DataFrame(data['data'], index=index, columns=columns)
    if data['is_series']:
        return df.iloc[:, 0]
    return df


def deserializer(obj):
    """
    Deserializer of numpy, pandas objects and dreye objects etc.
    """

    if len(obj) != 1:
        return obj

    for key, ele in obj.items():
        if key == DFRAME_PREFIX_PLUS:
            return _dataframe_deserializer(ele)
        elif key == RECARRAY_PREFIX:
            return pd.DataFrame(ele).to_records(index=False)
        elif key == ARRAY_PREFIX:
            return np.array(ele)
        elif key == DTYPE_PREFIX:
            return np.dtype(ele)
        elif key == PINT_PREFIX:
            return ureg(ele).units
        elif key == PICKLED_PREFIX:
            return spickleloads(codecs.decode(ele.encode(), "base64"))
        elif key == SERIES_PREFIX:
            return pd.Series(ele)
        elif key == DFRAME_PREFIX:
            return pd.DataFrame(ele)
        elif key == QUANT_PREFIX:
            return ele[0] * ele[1]
        elif key == DREYE_PREFIX:
            cls = _class_deserializer(ele)
            data = ele[DATA_PREFIX]
            if hasattr(cls, "_deprecated_kws"):
                for key_old, key_new in cls._deprecated_kws.items():
                    if key_old in data:
                        if key_new is None:
                            data.pop(key_old)
                        else:
                            data[key_new] = data.pop(key_old)
            return cls.from_dict(ele[DATA_PREFIX])
        elif key == DICTABLE_PREFIX:  # possibly custom classes
            cls = spickleloads(
                codecs.decode(ele[PICKLED_PREFIX].encode(), "base64")
            )
            return cls.from_dict(ele[DATA_PREFIX])

    return obj


def spickledumps(obj):
    """
    Saver pickle.dumps for relative imports
    """
    # TODO doesn't work across unix and windows together
    # TODO doesn't work properly if sys path was modified!
    module = inspect.getmodule(obj)
    # if isnstance
    if module is None:
        module = inspect.getmodule(type(obj))

    # package name
    package_name = module.__name__.split('.')[0]
    # __main__ handled separately
    if package_name == '__main__':
        return b"\0" + cloudpickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    # cwd and modulepath
    cwd = os.getcwd()
    modulepath = module.__file__
    relativepath = os.path.split(modulepath)[-1]
    # assume import of python folder from working directory
    if modulepath.startswith(cwd) and os.path.exists(package_name):
        folder_data = {}
        for root, d_names, f_names in os.walk(package_name):
            folder_name = os.path.split(root)[-1]
            if (
                folder_name in ignore.ignore_folder
                or folder_name.endswith(ignore.ignore_folder_endings)
                or folder_name.endswith(ignore.ignore_folder_starts)
            ):
                # skip unnecessary folders
                continue
            # save only allowed files
            f_names = [
                name for name in f_names
                if not (
                    name in ignore.ignore_file
                    or name.endswith(ignore.ignore_file_endings)
                    or name.startswith(ignore.ignore_file_starts)
                )
            ]
            for f_name in f_names:
                f_path = os.path.join(root, f_name)

                with open(f_path, "r") as f:
                    contents = f.read()

                folder_data[f_path] = contents
        folder_bytes = pickle.dumps(
            folder_data, protocol=pickle.HIGHEST_PROTOCOL
        )
        data_bytes = cloudpickle.dumps(
            obj, protocol=pickle.HIGHEST_PROTOCOL
        )
        return b"\1" + pickle.dumps(
            (folder_bytes, data_bytes),
            protocol=pickle.HIGHEST_PROTOCOL
        )

    # assume import of python file from working directory
    elif modulepath.startswith(cwd) and os.path.exists(relativepath):
        with open(relativepath, "r") as f:
            contents = f.read()

        folder_bytes = pickle.dumps(
            {relativepath: contents}, protocol=pickle.HIGHEST_PROTOCOL
        )
        data_bytes = cloudpickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        return b"\1" + pickle.dumps(
            (folder_bytes, data_bytes),
            protocol=pickle.HIGHEST_PROTOCOL
        )
    else:
        # just use pickledumps
        return b"\0" + cloudpickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def spickleloads(binary):
    """
    Saver pickle.loads for relative imports
    """

    number = binary[0]
    try:
        data = pickle.loads(binary[1:])
    except AttributeError:
        data = pandasloads(binary[1:])

    if not number:
        return data

    elif number == 1:
        folder_data = pickle.loads(data[0])
        for f_path, contents in folder_data.items():
            f_path = os.path.normpath(f_path)
            # create files in cwd if they do not exist
            if os.path.exists(f_path):
                continue

            root, f_name = os.path.split(f_path)

            # create root directories
            if not os.path.exists(root):
                os.makedirs(root)

            # write files
            with open(f_path, "w") as f:
                f.write(contents)

        return pickle.loads(data[1])

    else:
        raise DreyeError(f"Unknown byte identifier number: '{number}'.")


def write_pickle(filename, data):
    """
    Write pickled file.

    Parameters
    ----------
    filename : str
        location of file.
    data : dict
        Data to write to PICKLE file.
    """

    compression = infer_compression_from_filename(filename)
    mode = "wb"
    if compression is None or compression == "pickle":
        file = open(filename, mode=mode)
    elif compression == "gzip":
        file = gzip.open(filename, mode=mode)
    elif compression == "bz2":
        file = bz2.open(filename, mode=mode)
    elif compression == "lzma":
        file = lzma.open(filename, mode=mode)

    with file as f:
        f.write(spickledumps(data))


def read_pickle(filename):
    """
    Read pickled file.

    Parameters
    ----------
    filename : str
        location of file.

    Returns
    -------
    data : dict
        Data in PICKLE file.
    """

    compression = infer_compression_from_filename(filename)
    mode = 'rb'
    if compression is None or compression == "pickle":
        file = open(filename, mode=mode)
    elif compression == "gzip":
        file = gzip.open(filename, mode=mode)
    elif compression == "bz2":
        file = bz2.open(filename, mode=mode)
    elif compression == "lzma":
        file = lzma.open(filename, mode=mode)

    with file as f:
        data = spickleloads(f.read())

    return data
