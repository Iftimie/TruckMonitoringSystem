from collections import Callable


#TODO try to annotate like this
# Callable[[type1, type2], {'arg3': int, 'arg4': Optional[int]}, ret]
def func_return_dict(arg1, arg2) -> dict:
    return {arg1: arg2}


def func_return_val():
    return 10


def complex_func(identifier, int_arg, str_arg, file_arg, func_arg: Callable, func_arg_ret_dict: Callable) -> dict:
    ret_dict = func_arg_ret_dict("func_arg_ret_dict_key", 10)
    assert ret_dict is not None
    val = "{},{},{},{}".format(identifier, int_arg, str_arg, func_arg(), file_arg.name)
    ret_dict = {"val": val}
    return ret_dict