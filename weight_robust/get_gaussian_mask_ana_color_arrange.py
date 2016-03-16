import json
import sys
import cv2
import numpy as np
from sklearn import preprocessing
import os

json_file = sys.argv[1]
img_size = (5, 500)
img_size = (6, 501)
output_dir = json_file + '.result' # sys.argv[2]
os.mkdir(output_dir)

def im_scale(X):
    va = np.sum(X)
    vb = np.sum(np.int32(X>0))
    vc = va/vb
    min_v = np.min(X)
    max_v = np.max(X)
    if min_v == max_v:
        return X, vc
    min_max_scaler = preprocessing.MinMaxScaler()
    shape = X.shape
    X = min_max_scaler.fit_transform(X.reshape(shape[0]*shape[1]))
    X = np.reshape(X, (shape[0], shape[1]))
    return X, vc

def im_relu(X):
    return X * (X>0)


def draw_map(X, shape, file_name):
    print 'Writting picture %s.' % file_name
    X = np.reshape(X, shape)
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

def arrange(img_list, row, col, pad=0):
    h,w,c = img_list[0].shape
    I = np.ones([row*(h+pad)-pad, col*(w+pad)-pad, 3])
    for i in range(row):
        for j in range(col):
            I[i*(h+pad):i*(h+pad)+h, j*(w+pad):j*(w+pad)+w, :] = img_list[i*col+j]
    return I

json_obj = json.loads(open(json_file).read())
cross_node = get_node(json_obj, "cross")
conv2_node = get_node(json_obj, "conv2")
pool2_node = get_node(json_obj, "pool2")
fc2_node = get_node(json_obj, "fc2")
label_node = get_node(json_obj, "label")

# img_size = 28
row_count = 18
col_count = 1

order = sorted( [(idx, v) for idx, v in enumerate(fc2_node[:,0,0,0])], key = lambda x: x[1], reverse = True )
order = sorted( [(od, v) for od, v in enumerate(order)], key = lambda x: x[1][0] )

for i in range(cross_node.shape[0]):
    img_list = []
    for j in range(cross_node.shape[1]):
        print 'cross processing %s %s' % (i, j)
        Y = cross_node[i][j]
        T = np.zeros([img_size[0], img_size[1], 3])
        T[:Y.shape[0],:Y.shape[1],0] = Y
        T[:Y.shape[0],:Y.shape[1],1] = Y
        T[:Y.shape[0],:Y.shape[1],2] = Y
        img_list.append(T)
    
    for j in range(conv2_node.shape[1]):
        print 'conv2 processing %s %s' % (i, j)
        Y = conv2_node[i][j]
        X, vc = im_scale(Y)
        T = np.zeros([img_size[0], img_size[1], 3])
        T[:X.shape[0],:X.shape[1],0] = X
        T[:X.shape[0],:X.shape[1],1] = X
        T[:X.shape[0],:X.shape[1],2] = X
        img_list.append(T)

    for j in range(pool2_node.shape[1]):
        print 'pool2 processing %s %s' % (i, j)
        Y = pool2_node[i][j]
        X, vc = im_scale(Y)
        T = np.zeros([img_size[0], img_size[1], 3])
        T[:X.shape[0],:X.shape[1],0] = X
        T[:X.shape[0],:X.shape[1],1] = X
        T[:X.shape[0],:X.shape[1],2] = X
        img_list.append(T)

    T = np.zeros([img_size[0], img_size[1], 3])
    label_X = T + label_node[i][0][0][0]
    img_list.append(label_X)

    # I = arrange(img_list, row_count, col_count, 2)
    I = arrange(img_list, len(img_list), col_count, 2)
    file_name = output_dir + '/%d.result.jpg' % order[i][0]
    draw_map(I, I.shape, file_name)

print 'done'        

        
        
