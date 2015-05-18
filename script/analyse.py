#-*-coding:utf8-*-
import json
import numpy as np

def load_tensor(json_root):
    shape = json_root['data']['shape']
    len = shape[0]*shape[1]*shape[2]*shape[3]
    t = np.zeros(len)
    for i in range(len):
        t[i] = json_root['data']['value'][i]
    t = t.reshape(shape)
    return t

def concat_tensors(tensors):
    return np.concatenate(tensors, axis=0)

data_dir = '/home/wsx/exp/negneg/model/'
def getAllPoolRet():
    inFile = data_dir + 'test.cnn.ave.w2.d2.dp0.29900'
    outFile = inFile + '.pool_rep'
    batchs = json.loads(open(inFile).read())
    # print batchs
    pool_reps = concat_tensors([load_tensor(batch['pool_rep']) for batch in batchs])
    ys        = concat_tensors([load_tensor(batch['y']) for batch in batchs])
    fo = open(outFile, 'w')
    print 'ys.shape:', ys.shape
    print 'pool_rep.shape:' ,pool_reps.shape
    n_example = ys.shape[0]
    for i in range(n_example):
        fo.write(str(int(ys[i][0][0][0])))
        for j in range(pool_reps.shape[3]):
            fo.write(' ' + str(pool_reps[i][0][0][j]))
        fo.write('\n')
    fo.close()
    

getAllPoolRet()
