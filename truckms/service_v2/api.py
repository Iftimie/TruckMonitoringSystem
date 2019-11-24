from flask import Flask, Blueprint
import time
import threading


class P2PFlaskApp(Flask):

    def __init__(self, *args, **kwargs):
        super(P2PFlaskApp, self).__init__(*args, **kwargs)
        self.roles = []
        self._time_regular_funcs = []
        self._time_regular_thread = None
        self._time_interval = 10

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

    def register_time_regular_func(self, f):
        """
        Registers a callable that is called every self.time_interval seconds.
        The callable will not receive any arguments. If arguments are needed, make it a partial function or a class with
        __call__ implemented.
        """
        self.time_regular_funcs.append(f)



