#!/usr/bin/env python3.7

# rpc_server.py

# Fix missing module issue: ModuleNotFoundError: No module named 'SimpleXMLRPCServer'
#from SimpleXMLRPCServer import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCServer

import os

# Put in your server IP here
IP='0.0.0.0'
PORT=64001

server = SimpleXMLRPCServer((IP, PORT))

def server_receive_file(arg):
    with open('out_file.txt', "wb") as handle:
        handle.write(arg.data)
        return True

server.register_function(server_receive_file, 'server_receive_file')
print('Control-c to quit')
server.serve_forever()