#-*-coding:utf8-*-
import copy, os

from gen_conf_file import *
from dataset_cfg import *

t_l2 = 0.
t_lr = 0.
init_t = 0.0
f_gate_bias = 0.0
i_gate_bias = 0.0

def gen_match_lstm(d_mem, init, lr, dataset, l2):
    use_gru = True

    is_whole = False
    # print "ORC: left & right lstm share parameters"
    net = {}

    ds = DatasetCfg(dataset)
    g_filler    = gen_uniform_filler_setting(init)
    zero_filler = gen_zero_filler_setting()
    g_updater   = gen_adagrad_setting(lr = lr, l2 = l2, batch_size = ds.train_batch_size)

    g_layer_setting = {}
    g_layer_setting['no_bias'] = False

    g_layer_setting['w_filler'] = g_filler 
    g_layer_setting['b_filler'] = zero_filler
    g_layer_setting['w_g_filler'] = g_filler 
    g_layer_setting['b_g_filler'] = zero_filler
    g_layer_setting['w_c_filler'] = g_filler 
    g_layer_setting['b_c_filler'] = zero_filler
    g_layer_setting['w_updater'] = g_updater
    g_layer_setting['b_updater'] = g_updater
    g_layer_setting['w_g_updater'] = g_updater
    g_layer_setting['b_g_updater'] = g_updater
    g_layer_setting['w_c_updater'] = g_updater
    g_layer_setting['b_c_updater'] = g_updater

    net['net_name'] = 'match_lcs_toy'
    net['need_reshape'] = True
    net_cfg_train, net_cfg_valid, net_cfg_test = {}, {}, {}
    net['net_config'] = [net_cfg_train, net_cfg_valid, net_cfg_test]
    net_cfg_train["tag"] = "Train"
    net_cfg_train["max_iters"] = ds.train_max_iters * 50
    net_cfg_train["display_interval"] = ds.train_display_interval
    net_cfg_train["out_nodes"] = ['loss']
    net_cfg_valid["tag"] = "Valid"
    net_cfg_valid["max_iters"] = ds.valid_max_iters 
    net_cfg_valid["display_interval"] = ds.valid_display_interval 
    net_cfg_valid["out_nodes"] = ['loss']
    net_cfg_test["tag"]  = "Test"
    net_cfg_test["max_iters"]  = ds.test_max_iters 
    net_cfg_test["display_interval"] = ds.test_display_interval
    net_cfg_test["out_nodes"]  = ['loss']
    layers = []
    net['layers'] = layers

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['x', 'y']
    layer['layer_name'] = 'train_data'
    layer['layer_type'] = 86 
    layer['tag'] = ['Train']
    setting = {}
    layer['setting'] = setting
    setting['batch_size'] = ds.train_batch_size
    setting['shuffle'] = True
    setting['is_whole'] = is_whole
    setting['data_file'] = ds.train_data_file
    setting['max_doc_len'] = ds.max_doc_len
    setting['min_doc_len'] = ds.min_doc_len

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['x', 'y']
    layer['layer_name'] = 'valid_data'
    layer['layer_type'] = 86 
    layer['tag'] = ['Valid']
    setting = {}
    layer['setting'] = setting
    setting['batch_size'] = ds.valid_batch_size
    setting['is_whole'] = is_whole
    setting['data_file'] = ds.valid_data_file
    setting['max_doc_len'] = ds.max_doc_len
    setting['min_doc_len'] = ds.min_doc_len

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['x', 'y']
    layer['layer_name'] = 'test_data'
    layer['layer_type'] = 86 
    layer['tag'] = ['Test']
    setting = {}
    layer['setting'] = setting
    setting['is_whole'] = is_whole
    setting['batch_size'] = ds.test_batch_size
    setting['data_file'] = ds.test_data_file
    setting['max_doc_len'] = ds.max_doc_len
    setting['min_doc_len'] = ds.min_doc_len

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['x']
    layer['top_nodes'] = ['l_sentence', 'r_sentence']
    layer['layer_name'] = 'sentence_split'
    layer['layer_type'] = 20 
    setting = {}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['l_sentence', 'r_sentence']
    layer['top_nodes'] = ['word_match_matrix']
    layer['layer_name'] = 'match'
    layer['layer_type'] = 23 
    layer['setting'] = {'op':'xor', 'is_var_len':True}

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['word_match_matrix']
    layer['top_nodes'] = ['word_match_matrix_swap_1']
    layer['layer_name'] = 'word_match_matrix_swap_layer_1'
    layer['layer_type'] = 42 
    setting = {'axis1':1, 'axis2':2}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['word_match_matrix_swap_1']
    layer['top_nodes'] = ['word_match_matrix_output']
    layer['layer_name'] = 'word_match_matrix_swap_layer_2'
    layer['layer_type'] = 42
    setting = {'axis1':2, 'axis2':3}
    layer['setting'] = setting

    if not use_gru:
        layer = {}
        layers.append(layer) 
        layer['bottom_nodes'] = ['word_match_matrix_output']
        layer['top_nodes'] = ['match_matrix_lt2br'] # left top to bottom right
        layer['layer_name'] = 'lstm_d2_lt2br'
        layer['layer_type'] = 10005
        setting = copy.deepcopy(g_layer_setting)
        layer['setting'] = setting
        print "ORC: LSTM dim:", d_mem 
        setting['d_mem'] = d_mem
        setting['reverse'] = False
        setting['f_gate_bias_init'] = f_gate_bias
        setting['o_gate_bias_init'] = 0.0
        setting['i_gate_bias_init'] = i_gate_bias 
    else:
        layer = {}
        layers.append(layer) 
        layer['bottom_nodes'] = ['word_match_matrix_output']
        layer['top_nodes'] = ['match_matrix_lt2br'] # left top to bottom right
        layer['layer_name'] = 'lstm_d2_lt2br'
        # print "Use One Gate GRU."
        layer['layer_type'] = 10010
        setting = copy.deepcopy(g_layer_setting)
        layer['setting'] = setting
        print "ORC: LSTM dim:", d_mem 
        setting['d_mem'] = d_mem
        setting['reverse'] = False
        # print "Unuse reset gate"
        setting['is_use_reset_gate'] = False
        setting['f_gate_bias_init'] = f_gate_bias
        setting['o_gate_bias_init'] = 0.0
        setting['i_gate_bias_init'] = i_gate_bias 

    if is_whole:
        # 要把轴再换过来，因为dynamic poolin的时候，轴对不上
        layer = {}
        layers.append(layer) 
        layer['bottom_nodes'] = ['match_matrix_lt2br']
        layer['top_nodes'] = ['match_matrix_swap']
        layer['layer_name'] = 'swap_4_dpool_1'
        layer['layer_type'] = 42 
        setting = {'axis1':3, 'axis2':2}
        layer['setting'] = setting

        layer = {}
        layers.append(layer) 
        layer['bottom_nodes'] = ['match_matrix_swap']
        layer['top_nodes'] = ['match_matrix_output']
        layer['layer_name'] = 'swap_4_dpool_2'
        layer['layer_type'] = 42
        setting = {'axis1':2, 'axis2':1}
        layer['setting'] = setting
    else:
        layer = {}
        layers.append(layer) 
        layer['bottom_nodes'] = ['match_matrix_lt2br']
        layer['top_nodes'] = ['pool_rep_last']
        layer['layer_name'] = 'last_pooling'
        layer['layer_type'] = 10006 
        layer['setting'] = {'type':'last'}



    layer = {}
    layers.append(layer) 
    if is_whole:
        layer['bottom_nodes'] = ['match_matrix_output', 'y']
    else:
        layer['bottom_nodes'] = ['pool_rep_last','y']
    layer['top_nodes'] = ['loss']
    layer['layer_name'] = 'euclid_distance'
    layer['layer_type'] = 63
    setting = {}
    layer['setting'] = setting
    setting['delta'] = 1.
    setting['temperature'] = 1.

    return net

run = 1
l2 = 0.
idx = 0
for dataset in ['lcs_toy_v10_varlen']:
    for d_mem in [1]:
       for gate_bias in [-0.2]:
           i_gate_bias = -0.1
           f_gate_bias = gate_bias
           for init in [0.1, 0.03, 0.01]:
               for lr in [0.3, 0.2, 0.1, 0.05]:
                   net = gen_match_lstm(d_mem=d_mem,init=init,lr=lr,dataset=dataset,l2=l2)
                   net['log'] = 'log.match.lstmd2.{0}.d{1}.run{2}.{3}'.format\
                                (dataset, str(d_mem), str(run), str(idx))
                   net["save_model"] = {"file_prefix": "./model/model."+str(idx),"save_interval": 2000}
                   net["save_activation"] = [{"tag":"Test","file_prefix": \
                                              "./model/test."+str(idx), \
                                              "save_interval": 1000, \
                                              "save_nodes":["x","y", \
                                                            "word_match_matrix",\
                                                            # "match_matrix_output",\
                                                            "loss"], \
                                              "save_iter_num":1}]


                   gen_conf_file(net, '/home/wsx/exp/match/{0}/lstmd2/run.{1}/'.format(dataset,str(run)) + \
                                      'model.match.lstmd2.{0}.d{1}.run{2}.{3}'.format\
                                      (dataset, str(d_mem), str(run), str(idx)))
                   idx += 1
