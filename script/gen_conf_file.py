#-*-coding:utf8-*-
import json

def gen_zero_filter_setting():
    return {'init_type':0, 'value':0}
def gen_uniform_filter_setting(interval):
    return {'init_type':2, 'range':interval}

def gen_sgd_setting(lr, l2 = None, batch_size = 1):
    # assert not l2 or l2 == 0
    setting = {}
    setting['updater_type'] = 0
    setting['lr'] = lr
    setting['batch_size'] = batch_size
    if l2:
        setting['l2'] = l2
    return  setting

def gen_adadelta_setting(l2 = None, batch_size = 1, eps = None, rho = None, norm2=0.):
    setting = {}
    setting['updater_type'] = 4
    setting['batch_size'] = batch_size
    if l2:
        setting['l2'] = l2
    if eps:
        setting['eps'] = eps
    if rho:
        setting['rho'] = rho
    if norm2 > 0:
        setting['norm2'] = norm2
    return setting

def gen_adagrad_setting(lr, l2 = None, max_iter = -1, batch_size = 1, eps = None):
    setting = {}
    setting['updater_type'] = 1 
    setting['lr'] = lr
    setting['batch_size'] = batch_size
    if l2:
        setting['l2'] = l2
    if eps:
        setting['eps'] = eps
    if max_iter > 0:
        setting['max_iter'] = max_iter
    return  setting

def gen_conf_file(net_setting, out_file):
    fo = open(out_file, 'w')
    fo.write(json.dumps(net_setting, sort_keys=True, indent=2))
    fo.close()
    return


# gen_conf_file(None, "tmp")
# obj = [[1,2,3],123,123.123,'abc',{'key1':(1,2,3),'key2':(4,5,6)}]
# encodedjson = json.dumps(obj)
# print repr(obj)
# print encodedjson
# 
