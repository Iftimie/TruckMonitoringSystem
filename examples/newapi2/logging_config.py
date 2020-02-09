from logging.config import dictConfig


def configure_logger(name, level='WARNING'):
    # https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/

    format_string = "[%(processName)s %(threadName)s %(asctime)20s - %(name)s.%(funcName)s:%(lineno)s %(levelname)s] %(message)s"

    dictConfig({
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {'MYFORMATTER': {
            'format': format_string,
        }},
        'handlers': {
            'console': {
                'level': level,
                'class': 'logging.StreamHandler',
                'stream': "ext://sys.stdout",
                'formatter': 'MYFORMATTER'
            },
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
        },  # TODO maybe add an email handler
        '': {
            # 'level': level,
            'handlers': ['console', 'info_file_handler', 'error_file_handler', 'wsgi', 'werkzeug']
        },
        'root': {
            # 'level': level,
            'handlers': ['console', 'info_file_handler', 'error_file_handler', 'wsgi', 'werkzeug']
        },
    })
