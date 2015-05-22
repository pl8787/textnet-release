#-*-coding:utf8-*-
import copy, os

from gen_conf_file import *
from dataset_cfg import *

def gen_match_bilstm_sim(d_mem, init, lr, dataset, l2, lstm_norm2):
    # print "ORC: left & right lstm share parameters"
    is_share = False
    net = {}

    ds = DatasetCfg(dataset)
    g_filler    = gen_uniform_filter_setting(init)
    zero_filler = gen_zero_filter_setting()
    g_updater   = gen_adagrad_setting(lr = lr, l2 = 0., batch_size = ds.train_batch_size)
    zero_l2_updater   = gen_adagrad_setting(lr = lr, batch_size = ds.train_batch_size)

    g_layer_setting = {}
    g_layer_setting['no_bias'] = True
    g_layer_setting['w_filler'] = g_filler 
    g_layer_setting['u_filler'] = g_filler
    g_layer_setting['b_filler'] = zero_filler
    g_layer_setting['w_updater'] = g_updater
    g_layer_setting['u_updater'] = g_updater
    g_layer_setting['b_updater'] = g_updater

    net['net_name'] = 'match_bilstm_mlp'
    net['need_reshape'] = False
    net_cfg_train, net_cfg_valid, net_cfg_test = {}, {}, {}
    net['net_config'] = [net_cfg_train, net_cfg_valid, net_cfg_test]
    net_cfg_train["tag"] = "Train"
    net_cfg_train["max_iters"] = ds.train_max_iters 
    net_cfg_train["display_interval"] = ds.train_display_interval
    net_cfg_train["out_nodes"] = ['loss', 'acc']
    net_cfg_valid["tag"] = "Valid"
    net_cfg_valid["max_iters"] = ds.valid_max_iters 
    net_cfg_valid["display_interval"] = ds.valid_display_interval 
    net_cfg_valid["out_nodes"] = ['loss','acc']
    net_cfg_test["tag"] = "Test"
    net_cfg_test["max_iters"]  = ds.test_max_iters 
    net_cfg_test["display_interval"] = ds.test_display_interval
    net_cfg_test["out_nodes"]  = ['loss','acc']
    layers = []
    net['layers'] = layers

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['x', 'y']
    layer['layer_name'] = 'train_data'
    layer['layer_type'] = 71
    layer['tag'] = ['Train']
    setting = {}
    layer['setting'] = setting
    setting['batch_size'] = ds.train_batch_size
    setting['data_file'] = ds.train_data_file
    setting['max_doc_len'] = ds.max_doc_len
    setting['min_doc_len'] = ds.min_doc_len

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['x', 'y']
    layer['layer_name'] = 'valid_data'
    layer['layer_type'] = 71
    layer['tag'] = ['Valid']
    setting = {}
    layer['setting'] = setting
    setting['batch_size'] = ds.valid_batch_size
    setting['data_file'] = ds.valid_data_file
    setting['max_doc_len'] = ds.max_doc_len
    setting['min_doc_len'] = ds.min_doc_len

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['x', 'y']
    layer['layer_name'] = 'test_data'
    layer['layer_type'] = 71
    layer['tag'] = ['Test']
    setting = {}
    layer['setting'] = setting
    setting['batch_size'] = ds.test_batch_size
    setting['data_file'] = ds.test_data_file
    setting['max_doc_len'] = ds.max_doc_len
    setting['min_doc_len'] = ds.min_doc_len

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['x']
    layer['top_nodes'] = ['word_rep_seq']
    layer['layer_name'] = 'embedding'
    layer['layer_type'] = 21
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['embedding_file'] = ds.embedding_file
    setting['update_indication_file'] = ds.update_indication_file
    setting['feat_size'] = ds.d_word_rep
    setting['word_count'] = ds.vocab_size
    print "ORC: not use l2 for embedding"
    setting['w_updater'] = zero_l2_updater
    setting['w_filler'] = {}
    setting['w_filler']['init_type'] = 2
    setting['w_filler']['range'] = 0.14

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['word_rep_seq']
    layer['top_nodes'] = ['l_lstm_seq']
    layer['layer_name'] = 'l_lstm'
    layer['layer_type'] = 24
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['d_mem'] = d_mem
    setting['reverse'] = False
    setting['grad_norm2'] = lstm_norm2

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['l_lstm_seq']
    layer['top_nodes'] = ['l_pool_rep']
    layer['layer_name'] = 'l_wholePooling'
    layer['layer_type'] =  25 
    setting = {"pool_type":"last"}
    layer['setting'] = setting

    # layer = {}
    # layers.append(layer) 
    # layer['bottom_nodes'] = ['pool_rep']
    # layer['top_nodes'] = ['drop_rep']
    # layer['layer_name'] = 'dropout'
    # layer['layer_type'] =  13
    # setting = {'rate':ds.dp_rate}
    # layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['word_rep_seq']
    layer['top_nodes'] = ['r_lstm_seq']
    layer['layer_name'] = 'r_lstm'
    layer['layer_type'] = 24
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['d_mem'] = d_mem
    setting['reverse'] = True 
    setting['grad_norm2'] = lstm_norm2
    if is_share:
        print "ORC: share parameters."
        share_setting_w = {}
        share_setting_w['param_id'] = 0
        share_setting_w['source_layer_name'] = 'l_lstm'
        share_setting_w['source_param_id'] = 0
        share_setting_u = {}
        share_setting_u['param_id'] = 1
        share_setting_u['source_layer_name'] = 'l_lstm'
        share_setting_u['source_param_id'] = 1
        share_setting_b = {}
        share_setting_b['param_id'] = 2
        share_setting_b['source_layer_name'] = 'l_lstm'
        share_setting_b['source_param_id'] = 2
        setting['share'] = [share_setting_w, share_setting_u, share_setting_b]

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['r_lstm_seq']
    layer['top_nodes'] = ['r_pool_rep']
    layer['layer_name'] = 'r_wholePooling'
    layer['layer_type'] =  25 
    setting = {"pool_type":"first"}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['l_pool_rep', 'r_pool_rep']
    layer['top_nodes'] = ['bi_pool_rep']
    layer['layer_name'] = 'concat'
    layer['layer_type'] = 18
    setting = {}
    layer['setting'] = setting
    setting['bottom_node_num'] = 2
    setting['concat_dim_index'] = 3

    # layer = {}
    # layers.append(layer) 
    # layer['bottom_nodes'] = ['bi_pool_rep']
    # layer['top_nodes'] = ['bi_pool_rep_drop']
    # layer['layer_name'] = 'dropout'
    # layer['layer_type'] =  13
    # setting = {'rate':ds.dp_rate}
    # layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['bi_pool_rep']
    layer['top_nodes'] = ['l_sentence', 'r_sentence']
    layer['layer_name'] = 'sentence_split'
    layer['layer_type'] = 20 
    setting = {}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['l_sentence', 'r_sentence']
    layer['top_nodes'] = ['sim']
    layer['layer_name'] = 'match'
    layer['layer_type'] = 23 
    layer['setting'] = {'op':'cos'}
    layer['setting'] = {'is_var_len':False}

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['sim']
    layer['top_nodes'] = ['softmax_prob']
    layer['layer_name'] = 'softmax_fullconnect'
    layer['layer_type'] = 11 
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['num_hidden'] = ds.num_class
    # setting['no_bias'] = False
    setting['w_filler'] = zero_filler

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['softmax_prob', 'y']
    layer['top_nodes'] = ['loss']
    layer['layer_name'] = 'softmax_loss'
    layer['layer_type'] = 51 
    setting = {}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['softmax_prob', 'y']
    layer['top_nodes'] = ['acc']
    layer['layer_name'] = 'accuracy'
    layer['layer_type'] = 56 
    setting = {'topk':1}
    layer['setting'] = setting
    return net

run = 1
l2 = 0.
# for dataset in ['paper']:
for dataset in ['msrp']:
    for d_mem in [30]:
        idx = 0
        for init in [0.5, 0.3, 0.1]:
            for lr in [0.3, 0.1, 0.05, 0.03, 0.01]:
                for lstm_norm2 in [10000]:
                    net = gen_match_bilstm_sim(d_mem=d_mem, init=init, lr=lr, dataset=dataset, l2=l2, lstm_norm2=lstm_norm2)
                    net['log'] = 'log.match.bilstm_sim.{0}.d{1}.run{2}.{3}'.format\
                                 (dataset, str(d_mem), str(run), str(idx))
                    gen_conf_file(net, '/home/wsx/exp/match/{0}/bilstm_sim/run.{1}/'.format(dataset,str(run)) + \
                                       'model.match.bilstm_sim.{0}.d{1}.run{2}.{3}'.format\
                                       (dataset, str(d_mem), str(run), str(idx)))
                    idx += 1
