#-*-coding:utf8-*-
import copy, os

from gen_conf_file import *
from dataset_cfg import *

def gen_bilstm(d_mem, init, lr, dataset):
    print "ORC: left & right lstm share parameters"
    is_share = True
    net = {}
    # dataset = 'tb_fine'
    # dataset = 'mr'
    if dataset == 'mr':
        net['cross_validation'] = 10
    ds = DatasetCfg(dataset)
    g_filler = gen_uniform_filter_setting(init)
    zero_filler = gen_zero_filter_setting()
    # g_updater = gen_adadelta_setting()
    g_updater = gen_adagrad_setting(lr = lr, l2 = 0., batch_size = ds.train_batch_size)

    g_layer_setting = {}
    g_layer_setting['no_bias'] = True
    g_layer_setting['w_filler'] = g_filler 
    g_layer_setting['u_filler'] = g_filler
    g_layer_setting['b_filler'] = zero_filler
    g_layer_setting['w_updater'] = g_updater
    g_layer_setting['u_updater'] = g_updater
    g_layer_setting['b_updater'] = g_updater

    net['net_name'] = 'bilstm'
    net['need_reshape'] = True
    net_cfg_train, net_cfg_valid, net_cfg_test = {}, {}, {}
    net['net_config'] = [net_cfg_train, net_cfg_valid, net_cfg_test]
    net_cfg_train["tag"] = "Train"
    net_cfg_train["max_iters"] = (ds.n_train * 10)/ ds.train_batch_size 
    net_cfg_train["display_interval"] = (ds.n_train/ds.train_batch_size)/300
    net_cfg_train["out_nodes"] = ['acc']
    net_cfg_valid["tag"] = "Valid"
    net_cfg_valid["max_iters"] = int(ds.n_valid/ds.valid_batch_size) 
    net_cfg_valid["display_interval"] = (ds.n_train/ds.train_batch_size)/3
    net_cfg_valid["out_nodes"] = ['acc']
    net_cfg_test["tag"] = "Test"
    net_cfg_test["max_iters"] = int(ds.n_test/ds.test_batch_size) 
    net_cfg_test["display_interval"] = (ds.n_train/ds.train_batch_size)/3
    net_cfg_test["out_nodes"] = ['acc']
    layers = []
    net['layers'] = layers

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['y', 'x']
    layer['layer_name'] = 'train_data'
    layer['layer_type'] = 72
    layer['tag'] = ['Train']
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['batch_size'] = ds.train_batch_size
    setting['data_file'] = ds.train_data_file
    setting['max_doc_len'] = ds.max_doc_len

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['y', 'x']
    layer['layer_name'] = 'valid_data'
    layer['layer_type'] = 72
    layer['tag'] = ['Valid']
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['batch_size'] = ds.valid_batch_size 
    setting['data_file'] = ds.valid_data_file
    setting['max_doc_len'] = ds.max_doc_len

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['y', 'x']
    layer['layer_name'] = 'test_data'
    layer['layer_type'] = 72
    layer['tag'] = ['Test']
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['batch_size'] = ds.test_batch_size 
    setting['data_file'] = ds.test_data_file
    setting['max_doc_len'] = ds.max_doc_len

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['x']
    layer['top_nodes'] = ['word_rep_seq']
    layer['layer_name'] = 'embedding'
    layer['layer_type'] = 21
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['embedding_file'] = ds.embedding_file
    setting['feat_size'] = ds.d_word_rep
    setting['word_count'] = ds.vocab_size

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

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['l_lstm_seq']
    layer['top_nodes'] = ['l_pool_rep']
    layer['layer_name'] = 'l_wholePooling'
    layer['layer_type'] =  25 
    setting = {"pool_type":"last"}
    layer['setting'] = setting

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
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['bottom_node_num'] = 2
    setting['concat_dim_index'] = 3

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['bi_pool_rep']
    layer['top_nodes'] = ['drop_rep']
    layer['layer_name'] = 'dropout'
    layer['layer_type'] =  13
    setting = {'rate':ds.dp_rate}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['drop_rep']
    layer['top_nodes'] = ['softmax_ret']
    layer['layer_name'] = 'softmax_fullconnect'
    layer['layer_type'] = 11 
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['num_hidden'] = ds.num_class
    setting['w_filler'] = zero_filler

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['softmax_ret', 'y']
    layer['top_nodes'] = ['loss']
    layer['layer_name'] = 'softmax_activation'
    layer['layer_type'] = 51 
    setting = {}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['softmax_ret', 'y']
    layer['top_nodes'] = ['acc']
    layer['layer_name'] = 'accuracy'
    layer['layer_type'] = 56 
    setting = {'topk':1}
    layer['setting'] = setting

    return net

for dataset in ['mr', 'tb_fine', 'tb_binary']:
    for d_mem in [50, 75]:
        idx = 0
        for init in [0.3, 0.1, 0.03]:
            for lr in [0.3, 0.1, 0.03]:
                net = gen_bilstm(d_mem = d_mem, init = init, lr =lr, dataset=dataset)
                net['log'] = 'log.bilstm.last.{0}.d{1}.share.{2}'.format(dataset, str(d_mem), str(idx))
                # gen_conf_file(net, '/home/wsx/exp/tb/log/run.3/bilstm.max.tb_fine.model.' + str(idx))
                gen_conf_file(net, '/home/wsx/exp/gate/lstm/run.1/model.bilstm.last.{0}.d{1}.share.{2}'.format(dataset, str(d_mem), str(idx)))
                idx += 1
                # os.system("../bin/textnet ../bin/conv_lstm_simulation.model > ../bin/simulation/neg.gen.train.{0}".format(d_mem))
