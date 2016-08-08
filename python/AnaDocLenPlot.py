
# -*- coding = utf-8 -*-
import os
import sys
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def PlotTwoScatter(x1,x2,savefile):
    fig = plt.figure(figsize=(8,8),dpi=100)
    fig.text(0.85,0.85,"rel-doc",size=16,ha="center",va="center",color="#000000",weight="bold")
    fig.text(0.85,0.42,"ret-doc",size=16,ha="center",va="center",color="#000000",weight="bold")
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)
    xk = x1.keys()
    xk.extend(x2.keys())
    xmax = np.max(xk)
    print xmax
    ax1.set_xlim(1,xmax)
    ax2.set_xlim(1,xmax)
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    sum1 = np.sum(x1.values())
    sum2 = np.sum(x2.values())
    ax1.scatter(x1.keys(),x1.values(),color='#ff7700',edgecolor='#333333',s=40,marker='D',label='rel-doc')
    ax2.scatter(x2.keys(),x2.values(),color='#00ff77',edgecolor='#333333',s=40,marker='D',label='ret-doc')
    plt.savefig(savefile,dpi=100)


if __name__ == '__main__':
    ranklistfile = sys.argv[3]
    relfile = sys.argv[2]
    docinfofile = sys.argv[1]
    savefile = sys.argv[4]
    print 'doc-len file:',docinfofile
    print 'rel     file:',relfile
    print 'ranklist file:',ranklistfile
    ret = {} # query: [doc]  (top-20 docs id)
    with open(ranklistfile,'r') as f:
        for line in f:
            r = line.split()
            qid = r[0]
            did = r[2]
            rank = int(r[3])
            if qid in ret and rank > 20:
                continue
            if qid not in ret:
                ret[qid] = []
            ret[qid].append(did)
    rel = {} # query: [doc]  (relevance doc id)
    with open(relfile,'r') as f:
        for line in f:
            r = line.split()
            qid = r[0]
            did = r[2]
            label = int(r[3])
            if qid not in rel:
                rel[qid] = []
            if label > 0:
                rel[qid].append(did)
    doclen = {} # doc-id : length
    with open(docinfofile,'r') as f:
        for line in f:
            r = line.split()
            did = r[0]
            len = int(r[1])
            doclen[did] = len
    ret_len = []
    rel_len = []
    for qid,ds in ret.items():
        for d in ds:
            ret_len.append(doclen[d])
        for d in rel[qid]:
            if d in doclen:
                rel_len.append(doclen[d])
    #print ret_len
    #print rel_len
    rel_len_map = {}
    ret_len_map = {}
    for d in rel_len:
        idx = int(d / 100)
        if(idx > 200): 
            continue
        if idx not in rel_len_map:
            rel_len_map[idx] = 1
        else:
            rel_len_map[idx] += 1
        #print idx,',',
    #print '\n'
    for d in ret_len:
        idx = int(d / 100)
        if(idx > 200):   
            continue
        if idx not in ret_len_map:
            ret_len_map[idx] = 1
        else:
            ret_len_map[idx] += 1
        #print idx,',',
    #print '\n'
    PlotTwoScatter(rel_len_map,ret_len_map,savefile)



