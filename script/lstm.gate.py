#-*-coding:utf8-*-
import copy, os

from gen_conf_file import *
from dataset_cfg import *

def gen_gate_lstm(d_mem, init, lr, dataset):
    # print "ORC: left & right lstm share parameters"
    is_share = False
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

    net['net_name'] = 'gate_lstm'
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
    layer['top_nodes'] = ['drop_rep']
    layer['layer_name'] = 'dropout'
    layer['layer_type'] =  13
    setting = {'rate':ds.dp_rate}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['drop_rep']
    layer['top_nodes'] = ['pos_score']
    layer['layer_name'] = 'dim_reduction_for_softmax'
    layer['layer_type'] = 28
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['num_hidden'] = ds.num_class 

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['pos_score']
    layer['top_nodes'] = ['pos_prob']
    layer['layer_name'] = 'softmax_func'
    layer['layer_type'] = 37 # softmax_func
    setting = {"axis":3}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['drop_rep']
    layer['top_nodes'] = ['pos_weight_score']
    layer['layer_name'] = 'dim_reduction_for_weight'
    layer['layer_type'] = 28
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['num_hidden'] = 1

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['pos_weight_score']
    layer['top_nodes'] = ['pos_weight_prob']
    layer['layer_name'] = 'weight_softmax'
    layer['layer_type'] = 38 # softmax_func_var_len
    setting = {}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['pos_weight_prob', 'pos_prob']
    layer['top_nodes'] = ['pos_prob_reweight']
    layer['layer_name'] = 'weight_product'
    layer['layer_type'] = 35 
    setting = {}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['pos_prob_reweight']
    layer['top_nodes'] = ['final_prob']
    layer['layer_name'] = 'sum'
    layer['layer_type'] = 39 
    setting = {"axis":2}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['final_prob', 'y']
    layer['top_nodes'] = ['loss']
    layer['layer_name'] = 'cross_entropy'
    layer['layer_type'] = 57 
    setting = {}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['final_prob', 'y']
    layer['top_nodes'] = ['acc']
    layer['layer_name'] = 'accuracy'
    layer['layer_type'] = 56 
    setting = {'topk':1}
    layer['setting'] = setting

    return net

for dataset in ['tb_binary']:
    for d_mem in [50]:
        idx = 0
        for init in [0.03]:
            # for lr in [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]:
            for lr in [0.03]:
                net = gen_gate_lstm(d_mem = d_mem, init = init, lr =lr, dataset=dataset)
                net['log'] = 'log.lstm.max.{0}.d{1}.{2}'.format(dataset, str(d_mem), str(idx))
                # gen_conf_file(net, '/home/wsx/exp/tb/log/run.3/bilstm.max.tb_fine.model.' + str(idx))
                gen_conf_file(net, '/home/wsx/exp/gate/lstm/run.11/model.lstm.gate.{0}.d{1}.{2}'.format(dataset, str(d_mem), str(idx)))
                idx += 1
                # os.system("../bin/textnet ../bin/conv_lstm_simulation.model > ../bin/simulation/neg.gen.train.{0}".format(d_mem))
