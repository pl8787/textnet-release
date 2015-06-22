#-*-coding:utf8-*-
import copy, os

from gen_conf_file import *
from dataset_cfg import *

def gen_lm_lstm_autoencoder(d_mem, init, lr, dataset, l2, lstm_norm2, negative_num):
    left_2_right = True
    if left_2_right:
        print "use -> LSTM"
    else:
        print "use <- lSTM"
    net = {}

    ds = DatasetCfg(dataset)
    g_filler        = gen_uniform_filter_setting(init)
    zero_filler     = gen_zero_filter_setting()
    g_updater       = gen_adagrad_setting(lr=lr, l2=l2, batch_size=ds.train_batch_size)
    zero_l2_updater = gen_adagrad_setting(lr=lr, batch_size=ds.train_batch_size)

    g_layer_setting = {}
    g_layer_setting['no_bias']   = True

    net['net_name'] = 'lm_lstm_autencoder'
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
    layer['top_nodes'] = ['x', 'y']
    layer['layer_name'] = 'train_data'
    layer['layer_type'] = 77
    layer['tag'] = ['Train']
    setting = {}
    layer['setting'] = setting
    setting['batch_size'] = ds.train_batch_size
    setting['data_file'] = ds.train_data_file
    setting['max_doc_len'] = ds.max_doc_len
    setting['vocab_size'] = ds.vocab_size

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['x', 'y']
    layer['layer_name'] = 'valid_data'
    layer['layer_type'] = 77
    layer['tag'] = ['Valid']
    setting = {}
    layer['setting'] = setting
    setting['batch_size'] = ds.train_batch_size
    setting['data_file'] = ds.valid_data_file
    setting['max_doc_len'] = ds.max_doc_len
    setting['vocab_size'] = ds.vocab_size

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['x', 'y']
    layer['layer_name'] = 'test_data'
    layer['layer_type'] = 77
    layer['tag'] = ['Test']
    setting = {}
    layer['setting'] = setting
    setting['batch_size'] = ds.train_batch_size
    setting['data_file'] = ds.test_data_file
    setting['max_doc_len'] = ds.max_doc_len
    setting['vocab_size'] = ds.vocab_size

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['x']
    layer['top_nodes'] = ['word_rep_seq']
    layer['layer_name'] = 'embedding'
    layer['layer_type'] = 21
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    # setting['embedding_file'] = ds.embedding_file
    # print "ORC: only update non exist word in w2v"
    # setting['update_indication_file'] = ds.update_indication_file
    setting['feat_size'] = ds.d_word_rep
    setting['word_count'] = ds.vocab_size
    print "ORC: not use l2 for embedding"
    setting['w_updater'] = zero_l2_updater
    setting['w_filler'] = g_filler 

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['word_rep_seq']
    layer['top_nodes'] = ['lstm_seq']
    layer['layer_name'] = 'lstm_autoencoder'
    layer['layer_type'] = 47
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['d_mem'] = ds.d_word_rep 
    # setting['grad_norm2'] = lstm_norm2
    # setting['grad_cut_off'] = 0.5
    setting['w_ec_filler']  = g_filler 
    setting['u_ec_filler']  = g_filler
    setting['b_ec_filler']  = zero_filler
    setting['w_dc_filler']  = g_filler 
    setting['u_dc_filler']  = g_filler
    setting['b_dc_filler']  = zero_filler
    setting['w_ec_updater'] = g_updater
    setting['u_ec_updater'] = g_updater
    setting['b_ec_updater'] = zero_l2_updater
    setting['w_dc_updater'] = g_updater
    setting['u_dc_updater'] = g_updater
    setting['b_dc_updater'] = zero_l2_updater
    if left_2_right:
        setting['reverse'] = False
    else:
        setting['reverse'] = True

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['lstm_seq']
    layer['top_nodes'] = ['lstm_seq_swap']
    layer['layer_name'] = 'swap_axis'
    layer['layer_type'] = 42
    layer['setting'] = {'axis1':1, 'axis2':2}

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['lstm_seq_swap', 'y']
    layer['top_nodes'] = ['class_prob', 'word_prob', 'final_prob', 'loss']
    layer['layer_name'] = 'word_class_loss'
    layer['layer_type'] = 59
    setting = {}
    layer['setting'] = setting
    setting['w_class_filler'] = zero_filler
    setting['b_class_filler'] = zero_filler
    setting['w_word_filler']  = zero_filler
    setting['b_word_filler']  = zero_filler
    setting['w_class_updater'] = g_updater
    setting['b_class_updater'] = zero_l2_updater
    setting['w_word_updater']  = g_updater
    setting['b_word_updater']  = zero_l2_updater
    setting['feat_size'] = d_mem
    setting['class_num'] = 1000
    setting['vocab_size'] = ds.vocab_size
    setting['word_class_file'] = ds.word_class_file 

    return net

run = 31
# l2 = 0.
for dataset in ['wiki']:
    for d_mem in [2]:
        idx = 0
        for init in [0.01]:
            for lr in [0.1, 0.03, 0.01, 0.003]:
                for negative_num in [20]:
                    for l2 in [0.000001]:
                        lstm_norm2 = 2
                        net = gen_lm_lstm_autoencoder(d_mem=d_mem, init=init, lr=lr, dataset=dataset, l2=l2, \
                                                      lstm_norm2=lstm_norm2, negative_num=negative_num)
                        net['log'] = 'log.lm_lstm_autoenoder.{0}.d{1}.run{2}.{3}'.format \
                                     (dataset, str(d_mem), str(run), str(idx))
                        net["save_model"] = {"file_prefix": "./model/model."+str(idx),"save_interval": 5000}
                        net["save_activation"] = [{"tag":"Valid","file_prefix": \
                                                   "./model/valid."+str(idx), \
                                                   "save_interval": 5000, \
                                                   "save_nodes":["x","y","lstm_seq","word_rep_seq"], \
                                                   "save_iter_num":1}]

                        gen_conf_file(net, '/home/wsx/exp/match/{0}_lm/run.{1}/'.format(dataset, str(run)) + \
                                           'model.lm_lstm_autoencoder.{0}.d{1}.run{2}.{3}'.format \
                                           (dataset, str(d_mem), str(run), str(idx)))
                        # gen_conf_file(net, '/home/wsx/exp/match/test/'.format(dataset, str(run)) + \
                        #                    'model.lm_lstm_autoencoder.{0}.d{1}.run{2}.{3}'.format \
                        #                    (dataset, str(d_mem), str(run), str(idx)))
                        idx += 1
