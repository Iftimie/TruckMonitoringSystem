from flask import Flask, Blueprint, make_response
import time
import threading
import logging
import requests
logger = logging.getLogger(__name__)
import io
from warnings import warn
import collections
import inspect


"""
https://kite.com/blog/python/functional-programming/
Using higher-order function with type comments is an advanced skill. Type signatures often become long and unwieldy 
nests of Callable. For example, the correct way to type a simple higher order decorator that returns the input function 
is by declaring F = TypeVar[‘F’, bound=Callable[..., Any]] then annotating as def transparent(func: F) -> F: return func
. Or, you may be tempted to bail and use Any instead of trying to figure out the correct signature.
"""


class P2PFlaskApp(Flask):
    """
    Flask derived class for P2P applications. In this framework, the P2P app can have different roles. Not all nodes in
    the network are equal. Some act only as clients, some are reachable and act as workers or brokers, some have GPUs
     but are not reachable and thus act as workers. Given this possible set of configurations, the P2PFlaskApp has a
     specific role (names such as "streamer", "bookkeeper", "worker", "broker", "clientworker" etc).
     Also given the specific role, the app may have or not a background task (for example a task that makes network
     discovery)
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            args: positional arguments
            overwritten_routes: list of tuples (route, function_pointer). In case of overwriting a blueprint route,
                we need to address this situation manually.
            kwargs: keyword arguments
        """
        self.overwritten_routes = []  # List[Tuple[str, callable]]
        super(P2PFlaskApp, self).__init__(*args, **kwargs)
        self.roles = []
        self._blueprints = {}
        self._time_regular_funcs = []
        self._time_regular_thread = None
        self._time_interval = 10

    def add_url_rule(self, rule, endpoint=None, view_func=None, **options):
        # Flask registers views when an application starts
        # do not add view from self.overwritten_routes
        for rule_, view_func_ in self.overwritten_routes:
            if rule_ == rule and view_func == view_func_:
                warn("Overwritten rule: {}".format(rule))
                return
        return super(P2PFlaskApp, self).add_url_rule(rule, endpoint, view_func, **options)

    @staticmethod
    def _time_regular(list_funcs, time_interval):
        while True:
            for f in list_funcs:
                f()
            time.sleep(time_interval)

    def register_time_regular_func(self, f):
        """
        Registers a callable that is called every self.time_interval seconds.
        The callable will not receive any arguments. If arguments are needed, make it a partial function or a class with
        __call__ implemented.
        """
        self._time_regular_funcs.append(f)

    def register_blueprint(self, blueprint, **options):
        if isinstance(blueprint, P2PBlueprint):
            for f in blueprint.time_regular_funcs:
                self.register_time_regular_func(f)
            self.roles.append(blueprint.role)
            self.overwritten_routes += blueprint.overwritten_rules

        self._blueprints[blueprint.name] = blueprint
        super(P2PFlaskApp, self).register_blueprint(blueprint)

    # TODO I should also implement the shutdown method that will close the time_regular_thread

    def run(self, *args, **kwargs):
        self._time_regular_thread = threading.Thread(target=P2PFlaskApp._time_regular,
                                                     args=(self._time_regular_funcs, self._time_interval))
        self._time_regular_thread.start()
        super(P2PFlaskApp, self).run(*args, **kwargs)


class P2PBlueprint(Blueprint):
    """
    The P2PBlueprint also has a background task, a function that is designed to be called every N seconds.

    Or should it be named something like ActiveBlueprint? And the Blueprint should have an alias such as PassiveBlueprint?
    """

    def __init__(self, *args, role, **kwargs):
        super(P2PBlueprint, self).__init__(*args, **kwargs)
        self.time_regular_funcs = []
        self.role = role
        self.rule_mappings = {}
        self.overwritten_rules = [] # List[Tuple[str, callable]]
        self.route("/echo", methods=['GET'])(P2PBlueprint.echo)

    @staticmethod
    def echo():
        """
        Implicit function binded(routed) to /echo path. This helps a client determine if a broker/worker/P2Papp if it still exists.
        """
        return make_response("I exist", 200)

    def register_time_regular_func(self, f):
        """
        Registers a callable that is called every self.time_interval seconds.
        The callable will not receive any arguments. If arguments are needed, make it a partial function or a class with
        __call__ implemented.
        """
        self.time_regular_funcs.append(f)

    def route(self, *args, **kwargs):
        """
        Overwritten method for catching the rules and their functions in a map. In case the function is a locally declared function such as a partial,
        and we may want to overwrite that method, we need to store somewhere that locally declared function, otherwise we cannot access it.

        Example of route override:
        https://github.com/Iftimie/TruckMonitoringSystem/blob/6405f0341ad41c32fae7e4bab2d264b65a1d8ee9/truckms/service/worker/broker.py#L163
        """
        if args:
            rule = args[0]
        else:
            rule = kwargs['rule']

        decorator_function = super(P2PBlueprint, self).route(*args, **kwargs)
        def decorated_function_catcher(f):
            if rule in self.rule_mappings:
                self.overwritten_rules.append((rule, self.rule_mappings[rule]))
            self.rule_mappings[rule] = f
            return decorator_function(f)

        return decorated_function_catcher


def self_is_reachable(local_port):
    """
    return the public address: ip:port if it is reachable
    It makes a call to a public server such as 'http://checkip.dyndns.org/'. Inspired from the bitcoin protocol
    else it returns None
    """
    externalipres = requests.get('http://checkip.dyndns.org/')
    part = externalipres.content.decode('utf-8').split(": ")[1]
    ip_ = part.split("<")[0]
    try:
        echo_response = requests.get('http://{}:{}/echo'.format(ip_, local_port), timeout=2)
        if echo_response.status_code == 200:
            return "{}:{}".format(ip_, local_port)
        else:
            return None
    except:
        return None


def find_update_callables(kwargs):
    callables = [v for v in kwargs.values() if isinstance(v, collections.Callable)]
    update_callables = list(filter(
        lambda c: ("return" in inspect.getsource(c) or 'lambda' in inspect.getsource(c)) and inspect.signature(
            c).return_annotation == dict, callables))
    return update_callables


def validate_arguments(args, kwargs):
    if len(args) != 0:
        raise ValueError("All arguments to a function in this p2p framework need to be specified keyword arguments")
    if "identifier" not in kwargs:
        raise ValueError("In this p2p framework, identifier must be passed as keyword argument. "
                         "This helps for memoization and retrieving the results from a function")
    for v in kwargs.values():
        if any(isinstance(v, T) for T in [dict, tuple]):
            raise ValueError("Currently, for simplicity only integers, floats, strings and a single file are allowed as "
                             "arguments")
    if "value_for_key_is_file" in kwargs.values():
        raise ValueError("'value_for_key_is_file' string is a reserved value in this p2p framework. It helps "
                         "identifying a file.")
    files = [v for v in kwargs.values() if isinstance(v, io.IOBase)]
    if len(files) > 1:
        raise ValueError("p2p framework does not currently support sending more files")
    if files:
        if any(not file.closed for file in files):
            raise ValueError("all files should be closed. I don't want to cause pain...")

    update_callables = find_update_callables(kwargs)
    if len(update_callables) > 1:
        raise ValueError("Only one function that has return_annotation with dict and has return keyword is accepted")
    warn(
        "If function returns a dictionary in order to update a document in collection, then annotate it with '-> dict'"
        "If found return value and dict return annotation. The returned value will be used to update the document in collection")
