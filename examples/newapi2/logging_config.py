from logging.config import dictConfig


def configure_logger(name, module_level_list=None, default_level='WARNING'):
    # https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/

    if module_level_list is None:
        module_level_list = []

    format_string = "[%(processName)s %(threadName)s %(asctime)20s - %(name)s.%(funcName)s:%(lineno)s %(levelname)s] %(message)s"

    handlers = {
        "info_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "MYFORMATTER",
            "filename": "{}_info.log".format(name),
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },
        "error_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "MYFORMATTER",
            "filename": "{}_errors.log".format(name),
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },
        'wsgi': {
            "level": "ERROR",
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'MYFORMATTER'
        },
        'werkzeug': {
            'level': "ERROR",
            'class': 'logging.StreamHandler',
            'stream': "ext://sys.stdout",
            'formatter': 'MYFORMATTER'
        }
    }  # TODO maybe add an email handler
    for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        handlers['console_'+level] = {'level': level,
                                      'class': 'logging.StreamHandler',
                                      'stream': "ext://sys.stdout",
                                      'formatter': 'MYFORMATTER'}
    dconf = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {'MYFORMATTER': {
            'format': format_string,
        }},
        'handlers': handlers,

        'loggers': {
            '': {
                'level': default_level,
                'handlers': ['console_'+default_level, 'info_file_handler', 'error_file_handler', 'wsgi', 'werkzeug']
            },
            'root': {
                'level': default_level,
                'handlers': ['console_'+default_level, 'info_file_handler', 'error_file_handler', 'wsgi', 'werkzeug']
            },
        },
    }
    # I do this like that because I want in console to ignore INFO messages from flask (root is WARNING)
    # but I do not want to ignore from other modules
    for m_l in module_level_list:
        dconf['loggers'][m_l[0].__name__] = {'level': m_l[1], 'handlers': ['console_'+m_l[1], 'info_file_handler', 'error_file_handler']}

    dictConfig(dconf)
