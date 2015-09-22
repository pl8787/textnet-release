import zmq
import json
import time
import sys

def get_nodes(model):
    node_set = set()
    node_list = []
    for layer in model['layers']:
        if layer['bottom_nodes']:
            for node in layer['bottom_nodes']:
                if node not in node_set:
                    node_set.add(node)
                    node_list.append(node)
        if layer['top_nodes']:
            for node in layer['top_nodes']:
                if node not in node_set:
                    node_set.add(node)
                    node_list.append(node)
    print node_list
    return node_list

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:5000")

node_list = get_nodes(json.loads(open(sys.argv[1]).read()))

data_file = open("sample.data","w")

#  Do 10 requests, waiting each time for a response
for request in range(1000):
    print("Sending request %s ..." % request)
    for node in node_list:
        socket.send_string(b'{"msg_type":2, "node_name": "%s", "static_node": ["diff"], "static_value": ["mean"]}' % node)

        #  Get the reply.
        message = socket.recv()
        print >> data_file, message
        message = json.loads(message)
        print("Received reply %s [ %s ]" % (request, message))

    time.sleep(1)

data_file.close()
