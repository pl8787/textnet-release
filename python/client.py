import zmq
import json
import time

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:5000")

#  Do 10 requests, waiting each time for a response
for request in range(1000):
    print("Sending request %s ..." % request)
    socket.send(b'{"msg_type":2, "node_name": "cross", "static_node": ["data"], "static_value": ["mean"]}')

    #  Get the reply.
    message = socket.recv()
    print("Received reply %s [ %s ]" % (request, message))

    time.sleep(5)
