from flask import Flask, make_response
import time
from truckms.service_v2.userclient.p2p_client import ServerThread
import requests
from threading import Thread

def delayed_shutdown(t):
    print("waiting before shutdown")
    time.sleep(1)
    print("shuttingdown")
    t.shutdown()
    print("shutdown")

app = Flask(__name__)
app.local_port = 5000
app.stop_background_threads = lambda : None
app.start_background_threads = lambda : None

@app.route("/")
def f():
    print("waiting before makresponse")
    time.sleep(10)
    print("making response")
    return make_response("asd")

t = ServerThread(app)
t.start()

t2 = Thread(target=delayed_shutdown, args=(t,))
t2.start()
try:
    res = requests.get("http://localhost:5000/")
    print(res)
except:
    print("exc")
t2.join()



"""

KILLING VERSION

from flask import Flask, make_response
import time
from truckms.service_v2.userclient.p2p_client import ServerThread
import requests
from threading import Thread
from multiprocessing import Process

def delayed_kill(p: Process):
    print("waiting before shutdown")
    time.sleep(1)
    print("shuttingdown")
    p.terminate()
    print("shutdown")

app = Flask(__name__)
app.local_port = 5000
app.stop_background_threads = lambda : None
app.start_background_threads = lambda : None

@app.route("/")
def f():
    print("waiting before makresponse")
    time.sleep(10)
    print("making response")
    return make_response("asd")


def start_server_subprocess():
    app.run("0.0.0.0", 5000)

p = Process(target=start_server_subprocess)
p.start()

t2 = Thread(target=delayed_kill, args=(p,))
t2.start()
try:
    res = requests.get("http://localhost:5000/")
    print(res)
except:
    print("exc")

"""