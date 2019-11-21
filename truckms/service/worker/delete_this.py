from flask import Flask, send_file
from truckms.service.bookkeeper import ServerThread

app = Flask(__name__)

@app.route("/download_file")
def download_file():
    with open("dummy_shit.txt") as f:
        f.write("dummy_shit")
    return send_file("dummy_shit.txt",)
# this is what I need
# https://stackoverflow.com/questions/45721350/python-flask-send-file-and-variable

