import json
import sys
import cv2
import numpy as np
from sklearn import preprocessing

json_file = sys.argv[1]
node_idx = int(sys.argv[2])
output_dir = sys.argv[3]

def draw_map(X, shape, file_name, relu = False):
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
  return 1.0 * sum / (shape[0]*shape[1])
  
def get_node(json_obj):
  node_data = json_obj[0]['data']
  node_shape = node_data["shape"]
  print 'get node shape: ', node_shape
  node_value = np.array(node_data["value"])
  node_value = np.reshape(node_value, node_shape)
  return node_value

json_obj = json.loads(open(json_file).read())

# list all layer names
while node_idx == -1:
  for idx, layer in enumerate(json_obj['config']['layers']):
    print idx,'\t',layer['layer_name']
  node_idx = input('node_idx:')

featmap_node = get_node(json_obj['layers_params'][node_idx])

for i in range(featmap_node.shape[0]):
    print 'processing %s' % (i)
    X = featmap_node[i]
    kernel_size = int(X.shape[0]**0.5)
    X = np.reshape(X, (kernel_size, kernel_size))
    e = get_energy(X, (X.shape[0], X.shape[1]))    
    file_name = '/'.join( [output_dir, \
      '%s_%s_%s.jpg'%(node_idx, i, e)] )
    draw_map(X, (kernel_size, kernel_size), file_name)

print 'done'    

    
    
