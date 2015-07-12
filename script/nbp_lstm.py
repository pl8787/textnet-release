#-*-coding:utf8-*-
import copy, os

from gen_conf_file import *
from dataset_cfg import *

def gen_nbp_lstm(d_mem, init, lr, dataset, l2, max_norm2, negative_num):
    net = {}

    ds = DatasetCfg(dataset)
    g_filler        = gen_uniform_filter_setting(init)
    zero_filler     = gen_zero_filter_setting()
    g_updater       = gen_sgd_setting(lr=lr, l2=l2, batch_size=ds.train_batch_size)
    zero_l2_updater = gen_sgd_setting(lr=lr, batch_size=ds.train_batch_size)

    g_layer_setting = {}
    g_layer_setting['no_bias'] = True

    net['net_name'] = 'nbp_lstm'
    net['need_reshape'] = False
    net_cfg_train, net_cfg_valid, net_cfg_test = {}, {}, {}
    net['net_config'] = [net_cfg_train, net_cfg_valid, net_cfg_test]
    net_cfg_train["tag"] = "Train"
    net_cfg_train["max_iters"] = ds.train_max_iters 
    net_cfg_train["display_interval"] = ds.train_display_interval
    net_cfg_train["out_nodes"] = ['loss','acc']
    net_cfg_valid["tag"] = "Valid"
    net_cfg_valid["max_iters"] = ds.valid_max_iters 
    net_cfg_valid["display_interval"] = ds.valid_display_interval 
    net_cfg_valid["out_nodes"] = ['acc']
    net_cfg_test["tag"] = "Test"
    net_cfg_test["max_iters"]  = ds.test_max_iters 
    net_cfg_test["display_interval"] = ds.test_display_interval
    net_cfg_test["out_nodes"]  = ['acc']
    layers = []
    net['layers'] = layers

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['u', 'c', 'c_len', 'y', 'ys' ]
    layer['layer_name'] = 'train_data'
    layer['layer_type'] = 73
    layer['tag'] = ['Train']
    setting = {}
    layer['setting'] = setting
    setting['batch_size'] = ds.train_batch_size
    setting['data_file'] = ds.train_data_file
    setting['max_session_len'] = ds.max_session_len
    setting['max_context_len'] = ds.max_context_len
    setting['train_or_pred'] = "train"

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['u', 'c', 'c_len', 'y', 'ys' ]
    layer['layer_name'] = 'valid_data'
    layer['layer_type'] = 73
    layer['tag'] = ['Valid']
    setting = {}
    layer['setting'] = setting
    setting['batch_size'] = ds.train_batch_size
    setting['data_file'] = ds.valid_data_file
    setting['max_session_len'] = ds.max_session_len
    setting['max_context_len'] = ds.max_context_len
    setting['train_or_pred'] = "pred"

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['u', 'c', 'c_len', 'y', 'ys' ]
    layer['layer_name'] = 'test_data'
    layer['layer_type'] = 73
    layer['tag'] = ['Test']
    setting = {}
    layer['setting'] = setting
    setting['batch_size'] = ds.train_batch_size
    setting['data_file'] = ds.test_data_file
    setting['max_session_len'] = ds.max_session_len
    setting['max_context_len'] = ds.max_context_len
    setting['train_or_pred'] = "pred"

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['u']
    layer['top_nodes'] = ['u_rep']
    layer['layer_name'] = 'user_embedding'
    layer['layer_type'] = 21
    setting = {}
    layer['setting'] = setting
    setting['feat_size']  = ds.d_user_rep
    setting['word_count'] = ds.num_user
    setting['w_updater'] = zero_l2_updater
    setting['w_filler'] = g_filler 

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['c']
    layer['top_nodes'] = ['c_rep']
    layer['layer_name'] = 'item_embedding'
    layer['layer_type'] = 21
    setting = {}
    layer['setting'] = setting
    setting['feat_size']  = ds.d_item_rep
    setting['word_count'] = ds.num_item
    setting['w_updater'] = zero_l2_updater
    setting['w_filler'] = g_filler 

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['c_rep']
    layer['top_nodes'] = ['c_rep_ave']
    layer['layer_name'] = 'session_ave_pool'
    layer['layer_type'] = 25
    setting = {'pool_type':'ave'}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['c_rep_ave', 'c_len']
    layer['top_nodes'] = ['c_rep_lstm_input']
    layer['layer_name'] = 'gen_lstm_input'
    layer['layer_type'] = 48
    layer['setting'] = {}

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['c_rep_lstm_input']
    layer['top_nodes'] = ['lstm_rep']
    layer['layer_name'] = 'lstm'
    layer['layer_type'] = 24
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['d_mem'] = d_mem
    setting['grad_norm2'] = 1000
    setting['max_norm2'] = max_norm2
    setting['grad_cut_off'] = 500
    setting['reverse'] = False
    setting['w_filler'] = g_filler
    setting['u_filler'] = g_filler
    setting['b_filler'] = zero_filler
    setting['w_updater'] = g_updater
    setting['u_updater'] = g_updater
    setting['b_updater'] = g_updater

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['lstm_rep']
    layer['top_nodes'] = ['lstm_rep_last']
    layer['layer_name'] = 'last_pool'
    layer['layer_type'] = 25
    setting = {'pool_type':'last'}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['lstm_rep_last', 'u_rep']
    layer['top_nodes'] = ['pred_rep']
    layer['layer_name'] = 'concat'
    layer['layer_type'] = 18
    setting = {'bottom_node_num':2, 'concat_dim_index':3}
    layer['setting'] = setting

    # layer = {}
    # layers.append(layer) 
    # layer['bottom_nodes'] = ['pred_rep']
    # layer['top_nodes'] = ['hidden_rep_linear']
    # layer['layer_name'] = 'transform'
    # layer['layer_type'] = 11 
    # setting = copy.deepcopy(g_layer_setting)
    # layer['setting'] = setting
    # setting['num_hidden'] = n_hidden

    # layer = {}
    # layers.append(layer) 
    # layer['bottom_nodes'] = ['hidden_rep_linear']
    # layer['top_nodes'] = ['hidden_rep_nonlinear']
    # layer['layer_name'] = 'activation'
    # layer['layer_type'] = 1 # relu 1 sigmoide 2 tanh 3
    # setting = {"phrase_type":2}
    # layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['pred_rep']
    layer['top_nodes'] = ['drop_rep']
    layer['layer_name'] = 'dropout'
    layer['layer_type'] =  13
    setting = {'rate':0}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['drop_rep']
    layer['top_nodes'] = ['softmax_ret']
    layer['layer_name'] = 'softmax_fullconnect'
    layer['layer_type'] = 11 
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['num_hidden'] = ds.num_item
    setting['w_filler'] = zero_filler
    setting['b_filler'] = zero_filler
    setting['w_updater'] = zero_l2_updater
    setting['b_updater'] = zero_l2_updater

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['softmax_ret', 'y']
    layer['top_nodes'] = ['loss']
    layer['layer_name'] = 'softmax_activation'
    layer['layer_type'] = 51 
    layer['setting'] = {}

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['softmax_ret', 'ys']
    layer['top_nodes'] = ['acc']
    layer['layer_name'] = 'accuracy'
    layer['layer_type'] = 56 
    setting = {'topk':5}
    layer['setting'] = setting

    return net

run = 5
# l2 = 0.
for dataset in ['tf']:
    for d_mem in [30]:
        idx = 0
        for init in [0.3, 0.1]:
            # for lr in [0.1, 0.03, 0.01, 0.003]:
            for lr in [0.3, 0.1, 0.03]:
                # for negative_num in [0]:
                for max_norm2 in [10, 1, 0.1]:
                    for l2 in [0.0]:
                        lstm_norm2 = 1000
                        net = gen_nbp_lstm(d_mem=d_mem, init=init, lr=lr, dataset=dataset, l2=l2, \
                                           max_norm2=max_norm2, negative_num=0)
                        net['log'] = 'log.nbp_lstm.{0}.d{1}.run{2}.{3}'.format \
                                     (dataset, str(d_mem), str(run), str(idx))
                        # net["save_model"] = {"file_prefix": "./model/model."+str(idx),"save_interval": 5000}
                        # net["save_activation"] = [{"tag":"Valid","file_prefix": \
                        #                            "./model/valid."+str(idx), \
                        #                            "save_interval": 5000, \
                        #                            "save_nodes":["x","y","lstm_seq","word_rep_seq"], \
                        #                            "save_iter_num":1}]

                        gen_conf_file(net, '/home/wsx/exp/nbp/{0}/run.{1}/'.format(dataset, str(run)) + \
                                           'model.nbp_lstm.{0}.d{1}.run{2}.{3}'.format \
                                           (dataset, str(d_mem), str(run), str(idx)))
                        # gen_conf_file(net, '/home/wsx/exp/match/test/'.format(dataset, str(run)) + \
                        #                    'model.lm_lstm_autoencoder.{0}.d{1}.run{2}.{3}'.format \
                        #                    (dataset, str(d_mem), str(run), str(idx)))
                        idx += 1
