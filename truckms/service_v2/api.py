from flask import Flask, Blueprint, make_response
import time
import threading
from typing import Tuple, List


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
        self._time_regular_funcs = []
        self._time_regular_thread = None
        self._time_interval = 10

    def add_url_rule(self, rule, endpoint=None, view_func=None, **options):
        # Flask registers views when an application starts
        # do not add view from self.overwritten_routes
        for rule_, view_func_ in self.overwritten_routes:
            if rule_ == rule and view_func == view_func_:
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
        super(P2PFlaskApp, self).register_blueprint(blueprint)

    # TODO I should also implement the shutdown method that will close the time_regular_thread

    def run(self, *args, **kwargs):
        self._time_regular_thread = threading.Thread(target=P2PFlaskApp._time_regular,
                                                     args=(self._time_regular_funcs, self._time_interval))
        self._time_regular_thread.start()
        super(P2PFlaskApp, self).run(*args, **kwargs)


def echo():
    return make_response("I exist", 200)


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
        self.route("/echo", methods=['GET'])(echo)

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



