from flask import Flask

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


def test_var_path_length():


    test_client = app.test_client()
    res = test_client.get("/dum/plm/wtf")
    pass
