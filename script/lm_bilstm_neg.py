#-*-coding:utf8-*-
import copy, os

from gen_conf_file import *
from dataset_cfg import *

def gen_lm_bilstm_mlp(d_mem, init, lr, dataset, l2, lstm_norm2, negative_num):
    net = {}

    ds = DatasetCfg(dataset)
    g_filler    = gen_uniform_filter_setting(init)
    zero_filler = gen_zero_filter_setting()
    g_updater   = gen_adagrad_setting(lr = lr, l2 = l2, batch_size = ds.train_batch_size)
    zero_l2_updater   = gen_adagrad_setting(lr = lr, batch_size = ds.train_batch_size)

    g_layer_setting = {}
    g_layer_setting['no_bias'] = True
    g_layer_setting['w_filler'] = g_filler 
    g_layer_setting['u_filler'] = g_filler
    g_layer_setting['b_filler'] = zero_filler
    g_layer_setting['w_updater'] = g_updater
    g_layer_setting['u_updater'] = g_updater
    g_layer_setting['b_updater'] = g_updater

    net['net_name'] = 'lm_bilstm'
    net['need_reshape'] = False
    net_cfg_train, net_cfg_valid, net_cfg_test = {}, {}, {}
    net['net_config'] = [net_cfg_train, net_cfg_valid, net_cfg_test]
    net_cfg_train["tag"] = "Train"
    net_cfg_train["max_iters"] = ds.train_max_iters 
    net_cfg_train["display_interval"] = ds.train_display_interval
    net_cfg_train["out_nodes"] = ['loss']
    net_cfg_valid["tag"] = "Valid"
    net_cfg_valid["max_iters"] = ds.valid_max_iters 
    net_cfg_valid["display_interval"] = ds.valid_display_interval 
    net_cfg_valid["out_nodes"] = ['loss']
    net_cfg_test["tag"] = "Test"
    net_cfg_test["max_iters"]  = ds.test_max_iters 
    net_cfg_test["display_interval"] = ds.test_display_interval
    net_cfg_test["out_nodes"]  = ['loss']
    layers = []
    net['layers'] = layers

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['x', 'position', 'sample', 'y']
    layer['layer_name'] = 'train_data'
    layer['layer_type'] = 74
    layer['tag'] = ['Train']
    setting = {}
    layer['setting'] = setting
    setting['batch_size'] = ds.train_batch_size
    setting['data_file'] = ds.train_data_file
    setting['max_doc_len'] = ds.max_doc_len
    setting['negative_num'] = negative_num
    setting['position_num'] = 3
    setting['vocab_size'] = ds.vocab_size
    setting['word_freq_file'] = ds.word_freq_file
    setting['sample_exp_factor'] = 1. 

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['x', 'position', 'sample', 'y']
    layer['layer_name'] = 'train_data'
    layer['layer_type'] = 74
    layer['tag'] = ['Valid']
    setting = {}
    layer['setting'] = setting
    setting['batch_size'] = ds.valid_batch_size
    setting['data_file'] = ds.valid_data_file
    setting['max_doc_len'] = ds.max_doc_len
    setting['negative_num'] = negative_num
    setting['position_num'] = 3
    setting['vocab_size'] = ds.vocab_size
    setting['word_freq_file'] = ds.word_freq_file
    setting['sample_exp_factor'] = 1. 

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['x', 'position', 'sample', 'y']
    layer['layer_name'] = 'train_data'
    layer['layer_type'] = 74
    layer['tag'] = ['Test']
    setting = {}
    layer['setting'] = setting
    setting['batch_size'] = ds.test_batch_size
    setting['data_file'] = ds.test_data_file
    setting['max_doc_len'] = ds.max_doc_len
    setting['negative_num'] = negative_num
    setting['position_num'] = 3
    setting['vocab_size'] = ds.vocab_size
    setting['word_freq_file'] = ds.word_freq_file
    setting['sample_exp_factor'] = 1. 

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['x']
    layer['top_nodes'] = ['word_rep_seq']
    layer['layer_name'] = 'embedding'
    layer['layer_type'] = 21
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['embedding_file'] = ds.embedding_file
    print "ORC: only update non exist word in w2v"
    setting['update_indication_file'] = ds.update_indication_file
    setting['feat_size'] = ds.d_word_rep
    setting['word_count'] = ds.vocab_size
    print "ORC: not use l2 for embedding"
    setting['w_updater'] = zero_l2_updater

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['sample']
    layer['top_nodes'] = ['sample_rep']
    layer['layer_name'] = 'embedding_sample'
    layer['layer_type'] = 21
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['feat_size'] = d_mem # this layer is the softmax parameter, not word representation
    setting['word_count'] = ds.vocab_size
    setting['w_updater'] = zero_l2_updater
    # setting['w_filler'] = {}
    # setting['w_filler']['init_type'] = 0
    setting['embedding_file'] = ds.embedding_file
    print "ORC: only update non exist word in w2v"
    setting['update_indication_file'] = ds.update_indication_file

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['word_rep_seq']
    layer['top_nodes'] = ['l_lstm_seq']
    layer['layer_name'] = 'l_lstm'
    layer['layer_type'] = 24
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['d_mem'] = d_mem
    setting['grad_norm2'] = lstm_norm2
    setting['reverse'] = False

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['word_rep_seq']
    layer['top_nodes'] = ['r_lstm_seq']
    layer['layer_name'] = 'r_lstm'
    layer['layer_type'] = 24
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['d_mem'] = d_mem
    setting['grad_norm2'] = lstm_norm2
    setting['reverse'] = True 

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['l_lstm_seq', 'r_lstm_seq', 'position']
    layer['top_nodes'] = ['pos_pred_rep']
    layer['layer_name'] = 'pos_pred_rep_layer'
    layer['layer_type'] = 46 
    layer['setting'] = {}

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['pos_pred_rep']
    layer['top_nodes'] = ['pos_pred_rep_trans']
    layer['layer_name'] = 'pred_rep_trans_layer'
    layer['layer_type'] = 28
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['num_hidden'] = d_mem

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['pos_pred_rep_trans', 'sample_rep', 'y']
    layer['top_nodes'] = ['softmax_prob', 'loss']
    layer['layer_name'] = 'negative_sample_loss'
    layer['layer_type'] = 58
    layer['setting'] = {}

    return net

run = 14
l2 = 0.
for dataset in ['wiki']:
    for d_mem in [50]:
        idx = 0
        for init in [0.1]:
            for lr in [0.03, 0.01, 0.003]:
                for negative_num in [10]:
                    for l2 in [0.0001, 0.001, 0.01]:
                        lstm_norm2 = 2
                        net = gen_lm_bilstm_mlp(d_mem=d_mem, init=init, lr=lr, dataset=dataset, l2=l2, \
                                                lstm_norm2=lstm_norm2, negative_num=negative_num)
                        net['log'] = 'log.lm_bilstm.{0}.d{1}.run{2}.{3}'.format \
                                     (dataset, str(d_mem), str(run), str(idx))
                        net["save_model"] = {"file_prefix": "./model/model."+str(idx),"save_interval": 2000}
                        net["save_activation"] = [{"tag":"Valid","file_prefix": \
                                                   "./model/valid."+str(idx), \
                                                   "save_interval": 2000, \
                                                   "save_iter_num":1}]
                        gen_conf_file(net, '/home/wsx/exp/match/{0}_lm/run.{1}/'.format(dataset, str(run)) + \
                                           'model.lm_bilstm.{0}.d{1}.run{2}.{3}'.format \
                                           (dataset, str(d_mem), str(run), str(idx)))
                        idx += 1
