import json
import sys
import cv2
import numpy as np
from sklearn import preprocessing

json_file = sys.argv[1]
node_name = sys.argv[2]
output_dir = sys.argv[3]

def draw_map(X, shape, file_name, relu = True):
  print 'Writting picture %s.' % file_name
  if relu:
    X = X * (X>0)
  min_max_scaler = preprocessing.MinMaxScaler()
  X = min_max_scaler.fit_transform(X.reshape(shape[0]*shape[1]))
  X = np.reshape(X, (shape[0], shape[1]))
  # X = cv2.resize(X, (100, 100), interpolation=cv2.INTER_NEAREST)
  cv2.imwrite(file_name, X*255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
  
def get_energy(X, shape, use_abs = False):
  if use_abs:
    sum = np.sum(np.abs(X))
  else:
    sum = np.sum(X)
  return 1.0 * sum / (shape[0] * shape[1])
  
def get_node(json_obj, node_name):
  node_data = json_obj[0][node_name]['data']
  node_shape = node_data["shape"]
  node_value = np.array(node_data["value"])
  node_value = np.reshape(node_value, node_shape)
  return node_value

json_obj = json.loads(open(json_file).read())
label_node = get_node(json_obj, "label")
fc2_node = get_node(json_obj, "fc2")
featmap_node = get_node(json_obj, node_name)

for i in range(featmap_node.shape[0]):
  label = label_node[i][0][0][0]
  pred = 0 if fc2_node[i][0][0][0] > fc2_node[i][1][0][0] else 1
  for j in range(featmap_node.shape[1]):
    print 'processing %s %s' % (i, j)
    X = featmap_node[i][j]
    e = get_energy(X, (X.shape[0], X.shape[1]))    
    file_name = '/'.join( [output_dir, \
      '%s_%s_%s_%s_%s_%s.jpg'%(node_name, i, j, e, label, pred)] )
    draw_map(X, (X.shape[0], X.shape[1]), file_name)

print 'done'    

    
    
