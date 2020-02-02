from collections import Callable
from typing import List, Dict
from io import IOBase
import inspect
from multipledispatch import dispatch
import dill
import base64
from warnings import warn

"""
Get comparable from value (especially important for Callable and IOBase)
"""

@dispatch(int)
def kicomp(value):
    return value


@dispatch(float)
def kicomp(value):
    return value


@dispatch(str)
def kicomp(value):
    return value


@dispatch(Callable)
def kicomp(value):
    return inspect.signature(value)


@dispatch(IOBase)
def kicomp(value):
    return value.name


"""
Get string serialized version of value (especially important for Callable and IOBase)
"""

@dispatch(int)
def cls_finder(value):
    return int


@dispatch(float)
def cls_finder(value):
    return float


@dispatch(str)
def cls_finder(value):
    return str


@dispatch(list)
def cls_finder(l):
    if len(l) == 0:
        return List
    else:
        assert all(type(l[0]) == type(v) for v in l)
        return List[cls_finder(l[0])]


@dispatch(Callable)
def cls_finder(value):
    return Callable


@dispatch(IOBase)
def cls_finder(value):
    return IOBase


@dispatch(type(None))
def cls_finder(value):
    return type(None)


@dispatch(dict)
def cls_finder(d):
    if len(d) == 0:
        return Dict
    else:
        keys = list(d.keys())
        values = list(d.values())
        assert all(type(values[0]) == type(v) for v in values)
        assert all(type(keys[0]) == type(v) for v in keys)
        return Dict[cls_finder(keys[0]), cls_finder(values[0])]


db_encoder = {int: lambda value: value,
              float: lambda value: value,
              str: lambda value: value,
              IOBase: lambda handle: handle.name,
              Callable: lambda func: base64.b64encode(dill.dumps(func)).decode('utf8'),
              List[str]: lambda value: value,
              List[int]: lambda value: value,
              List[float]: lambda value: value,
              List: lambda value: value,
              type(None): lambda value: None,
              Dict: lambda value: value,
              Dict[str, str]: lambda value: value}


"""
Get string deserialized version of value (especially important for Callable and IOBase)
Serialized int, float, str, bool values in this framework are still integers (they can be inserted and found in tinymongo as the original values)
If using another framework for storage and the serialized values would be strings, then another dispatch could be implemented as

@dispatch(str, int)
def kidser(ser_value, cls):
    return cls(ser_value)
"""

db_decoder = {int: lambda value: value,
              float: lambda value: value,
              str: lambda value: value,
              IOBase: lambda path: open(path, 'rb'),
              Callable: lambda value: dill.loads(base64.b64decode(value.encode('utf8'))),
              List[str]: lambda value: value,
              List[int]: lambda value: value,
              List[float]: lambda value: value,
              List: lambda value: value,
              type(None): lambda value: None,
              Dict: lambda value: value,
              Dict[str, str]: lambda value: value}


def serialize_doc_for_db(doc):
    serialized_doc = dict()
    for k, v in doc.items():
        serialized_doc[k] = db_encoder[cls_finder(v)](v)
    return serialized_doc


def deserialize_doc_from_db(doc, clsd):
    if clsd is None:
        warn("Document not deserialized from db")
        return doc
    deserialized_doc = {k:v for k, v in doc.items()}
    for k in clsd:
        deserialized_doc[k] = db_decoder[clsd[k]](doc[k])
    diff_keys = set(doc.keys()) - set(clsd.keys())
    if diff_keys:
        warn("The following keys do not have deserializers " + str(diff_keys))
    return deserialized_doc


def get_class_dictionary_from_doc(doc):
    return {k: cls_finder(v) for k, v in doc.items()}
