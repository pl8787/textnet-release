#-*-coding:utf8-*-
import copy, os

from gen_conf_file import *
from dataset_cfg import *


def gen_lstm(d_mem, init):
    net = {}
    dataset = 'simulation'
    ds = DatasetCfg(dataset)

    save_train, save_valid, save_test = {},{},{}
    save_model_setting = {}
    save_activation_setting = [save_train, save_valid, save_test]
    net['save_model'] = save_model_setting
    net['save_activation'] = save_activation_setting

    save_model_setting['save_interval'] = 100
    save_model_setting['file_prefix'] = './model/model.'
    save_train['tag'] = 'Train'
    save_train['file_prefix'] = './model/train.'
    save_train['save_interval'] = 100
    save_train['save_iter_num'] = 10
    save_valid['tag'] = 'Valid'
    save_valid['file_prefix'] = './model/valid.'
    save_valid['save_interval'] = 100
    save_valid['save_iter_num'] = 10
    save_test['tag'] = 'Test'
    save_test['file_prefix'] = './model/test.'
    save_test['save_interval'] = 100
    save_test['save_iter_num'] = 10


    g_filler = gen_uniform_filter_setting(init)
    zero_filler = gen_zero_filter_setting()
    g_updater = gen_adadelta_setting()

    g_layer_setting = {}
    g_layer_setting['no_bias'] = True
    g_layer_setting['phrase_type'] = 2
    g_layer_setting['w_filler'] = g_filler 
    g_layer_setting['u_filler'] = g_filler
    g_layer_setting['b_filler'] = zero_filler
    g_layer_setting['w_updater'] = g_updater
    g_layer_setting['u_updater'] = g_updater
    g_layer_setting['b_updater'] = g_updater

    net['net_name'] = 'lstm'
    net_cfg_train, net_cfg_valid, net_cfg_test = {}, {}, {}
    net['net_config'] = [net_cfg_train, net_cfg_valid, net_cfg_test]
    net_cfg_train["tag"] = "Train"
    net_cfg_train["max_iters"] = (ds.n_train * 10)/ ds.batch_size 
    net_cfg_train["display_interval"] = (ds.n_train/ds.batch_size)/30
    net_cfg_train["out_nodes"] = ['acc']
    net_cfg_valid["tag"] = "Valid"
    net_cfg_valid["max_iters"] = int(ds.n_valid/ds.batch_size) 
    net_cfg_valid["display_interval"] = (ds.n_train/ds.batch_size)/3
    net_cfg_valid["out_nodes"] = ['acc']
    net_cfg_test["tag"] = "Test"
    net_cfg_test["max_iters"] = int(ds.n_test/ds.batch_size) 
    net_cfg_test["display_interval"] = (ds.n_train/ds.batch_size)/3
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
    setting['phrase_type'] = 0
    setting['batch_size'] = ds.batch_size
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
    setting['phrase_type'] = 1
    setting['batch_size'] = ds.batch_size 
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
    setting['phrase_type'] = 1
    setting['batch_size'] = ds.batch_size 
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
    layer['top_nodes'] = ['lstm_seq']
    layer['layer_name'] = 'lstm'
    layer['layer_type'] = 24
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['d_mem'] = d_mem
    setting['reverse'] = False

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['lstm_seq']
    layer['top_nodes'] = ['pool_rep']
    layer['layer_name'] = 'wholePooling'
    layer['layer_type'] =  25 
    setting = {"phrase_type":2, "pool_type":"last"}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['pool_rep']
    layer['top_nodes'] = ['drop_rep']
    layer['layer_name'] = 'dropout'
    layer['layer_type'] =  13
    setting = {'phrase_type':2, 'rate':ds.dp_rate}
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
    setting = {'phrase_type':2}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['softmax_ret', 'y']
    layer['top_nodes'] = ['acc']
    layer['layer_name'] = 'accuracy'
    layer['layer_type'] = 56 
    setting = {'phrase_type':2, 'topk':1}
    layer['setting'] = setting

    return net

net = gen_lstm(d_mem = 10, init = 0.1) 
gen_conf_file(net, './test.savemodel.model')

