from collections import Callable
from typing import List, Dict
from io import IOBase
import inspect
from multipledispatch import dispatch
import dill
import base64
from warnings import warn
import binascii
import os
import varint
import mmh3
import struct


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


# TODO to compare contents create a Class that implements __==__ and received another object of same instance
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
    """
    A special case arrived when using '$or' in filter and the doc is actually a list
    For example a filter could be:
    {"$or":[{"filter_key1": "value2"},{"filter_key2": "value2"}]}
    """
    serialized_doc = dict()
    for k, v in doc.items():
        if isinstance(v, list) and all(isinstance(item, dict) for item in v):
            serialized_doc[k] = [serialize_doc_for_db(item) for item in v]
        else:
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


def get_class_dictionary_from_func(func):
    doc = inspect.signature(func).parameters
    return {k: v.annotation for k, v in doc.items()}


SAMPLE_THRESHOLD = 128 * 1024
SAMPLE_SIZE = 16 * 1024
def hashfileobject(f, sample_threshhold=SAMPLE_THRESHOLD, sample_size=SAMPLE_SIZE, hexdigest=False):
    #get file size from file object
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(0, os.SEEK_SET)

    if size < sample_threshhold or sample_size < 1:
        data = f.read()
    else:
        data = f.read(sample_size)
        f.seek(size//2)
        data += f.read(sample_size)
        f.seek(-sample_size, os.SEEK_END)
        data += f.read(sample_size)

    hash_tmp = mmh3.hash_bytes(data)
    hash_ = hash_tmp[7::-1] + hash_tmp[16:7:-1]
    enc_size = varint.encode(size)
    digest = enc_size + hash_[len(enc_size):]

    f.seek(0, os.SEEK_SET)

    return binascii.hexlify(digest).decode() if hexdigest else digest


bytes_hasher = {int: lambda value: mmh3.hash_bytes(struct.pack("i", value)),
                float: lambda value: mmh3.hash_bytes(struct.pack("f", value)),
                str: lambda value: mmh3.hash_bytes(bytes(value, encoding="utf-8")),
                IOBase: lambda handle: hashfileobject(handle),
                Callable: lambda func: mmh3.hash_bytes(dill.dumps(func)),
                List[str]: lambda lstr: mmh3.hash_bytes(b''.join([bytes(v, encoding="utf-8") for v in lstr])),
                List[int]: lambda lint: mmh3.hash_bytes(b''.join([struct.pack("i", v) for v in lint])),
                List[float]: lambda lfloat: mmh3.hash_bytes(b''.join([struct.pack("f", v) for v in lfloat])),
                List: lambda value: mmh3.hash_bytes(b''),
                type(None): lambda value: mmh3.hash_bytes(b''),
                Dict: lambda value: mmh3.hash_bytes(b''),
                Dict[str, str]: lambda value: mmh3.hash_bytes(b''.join([struct.pack("i", k+v) for k, v in value.items()]))}


def hash_kwargs(doc):
    acc = b''
    for k in sorted(doc.keys()):
        v = doc[k]
        acc += bytes_hasher[cls_finder(v)](v)
    hash_ = mmh3.hash_bytes(acc)
    return binascii.hexlify(hash_).decode()
