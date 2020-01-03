from flask import Flask
from flask import request
app = Flask(__name__)

# @app.route('/', defaults={'path': ''})
# @app.route('/<path:path>')
# def get_dir(path):
#     print("\n\n\n\n\n")
#     print(path)
#     return "aaa"

@app.route('/dum/<path:path>')
def get_dir(path):
    print("\n\n\n\n\n")
    print(path)
    return "aaa"

@app.route('/simplepath')
def simplepath():
    print(request.args['user'])


def test_var_path_length():


    test_client = app.test_client()
    res = test_client.get("/dum/plm/wtf")
    nested_dict = {"a": {"b":"c"},
                   "d": {"e":"f",
                         "g":"h"}}
    res = test_client.get("/simplepath", query_string=nested_dict)
    pass
