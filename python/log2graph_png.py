import sys
import re
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

reduce_interval = {
    "Train" : 1000,
    "Valid" : 1000,
    "Test" : 1000
        }

seperator = ','  # '\t'

def reduce_result(x, interval):
    y = {}
    count = {}
    for iter, value in x:
        idx = iter / interval
        if idx not in y:
            y[idx] = 0.0
            count[idx] = 0
        y[idx] += value
        count[idx] += 1
    for idx in y:
        y[idx] /= count[idx]

    # convert dict to list
    idx = 0
    rtn = []
    while True:
        if idx in y:
            rtn.append(y[idx])
            idx += 1
        elif idx == 0:
            rtn.append(0)
            idx += 1
        else:
            break
    return rtn

# test = '[Train:kTrain]\tIter\t3321:\tOut[loss] =\t0.228255'
pattern_raw = r"\[.+?\] \[(.+?)\:(.+?)\]\tIter\t(.+?)\:\tOut\[(.+?)\].*?=\t(.+)"

if __name__=='__main__':
    log_lines = {}

    for line in open(sys.argv[1]):
        m = re.match(pattern_raw, line)
        if m:
            tag = m.group(1)
            phase = m.group(2)
            iter = int(m.group(3))
            node = m.group(4)
            value = float(m.group(5))
            if tag not in log_lines:
                log_lines[tag] = {}
            if node not in log_lines[tag]:
                log_lines[tag][node] = []
            log_lines[tag][node].append([iter, value])
    #print log_lines['Train']
    outpufile = sys.argv[1] + '.png'
    
    loss_x = []
    loss_y = []
    for idx,k in enumerate(log_lines['Train']['loss']):
        if idx % 10 == 0:
            loss_x.append(k[0])
            loss_y.append(k[1])
    train_map_x = []
    train_map_y = []
    test_map_x = []
    test_map_y = []

    for k in log_lines['Valid']['MAP']:
        train_map_x.append(k[0])
        train_map_y.append(k[1])
    for k in log_lines['Test']['MAP']:
        test_map_x.append(k[0])
        test_map_y.append(k[1])

    fig = plt.figure(figsize = (8,8),dpi = 120)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    
    p1 = ax1.plot(loss_x,loss_y,'-o',color="#ff0000",linewidth=2,markersize=2,label='train loss')
    
    x = np.linspace(0,len(test_map_x),len(test_map_x))
    xticks = []
    max_test_map = np.max(test_map_y)
    max_test_map_x = 0
    for idx,k in enumerate(test_map_y):
        if k == max_test_map:
            max_test_map_x = idx
            break

    ax2.set_xticks(x)
    ax2.set_xticklabels(train_map_x,rotation=70)
    p2 = ax2.plot(x,train_map_y,'-o',color="#78ff45",linewidth=2,markersize=2,label='train map')
    p3 = ax2.plot(x,test_map_y,'-o',color="#7845ff",linewidth=2,markersize=2,label='test map')
    ax2.annotate('%f'%(max_test_map),xy=(max_test_map_x,max_test_map),xytext=(max_test_map_x,max_test_map+0.0001))
    plt.legend(loc=2)
    plt.savefig(outpufile,dpi=120)


