import json
import sys
import cv2
import numpy as np
from sklearn import preprocessing

pred_file = open('pred.txt', 'w')
label_file = open('label.txt', 'w')

file_list = [(int(f.split('.')[-1]), f) for f in sys.argv[1:]]
file_list = sorted(file_list, key = lambda x : x[0])

for i_part, f_name in file_list:
    print i_part
    json_file = f_name
    
    json_str = open(json_file).read()
    json_obj = json.loads(json_str)[0]
    
    node_data = json_obj["fc2"]["data"]
    label_data = json_obj["label"]["data"]["value"]
    
    node_shape = node_data["shape"]
    node_value = np.array(node_data["value"])
    
    node_value = np.reshape(node_value, node_shape)
    
    for i in range(node_shape[0]):
        print 'processing %s' % i
        
        X = node_value[i]
        if X[0]>X[1]:
            print >>pred_file, 0
        else:
            print >>pred_file, 1
    
        print >>label_file, int(label_data[i])
pred_file.close()
label_file.close()
