### rpc_client.py
#!/usr/bin/env python3.7

import os


# client.py
import sys

# The answer is that the module xmlrpc is part of python3

import xmlrpc.client

#Put your server IP here
IP='localhost'
PORT=64001


url = 'http://{}:{}'.format(IP, PORT)
###server_proxy = xmlrpclib.Server(url)
client_server_proxy = xmlrpc.client.ServerProxy(url)

curDir = os.path.dirname(os.path.realpath(__file__))
filename = __file__

with open(filename, "rb") as handle:
    binary_data = xmlrpc.client.Binary(handle.read())
    client_server_proxy.server_receive_file(binary_data)