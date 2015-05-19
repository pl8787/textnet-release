#-*-coding:utf8-*-
import copy, os

from gen_conf_file import *


def gen_rnn(d_mem, init_iterval):
    dataset = 'mr'
    # dataset = 'simulation'
    if dataset == 'mr':
        train_data_file = '/home/wsx/dl.shengxian/data/mr/lstm.train.nopad'
        valid_data_file = '/home/wsx/dl.shengxian/data/mr/lstm.dev.nopad'
        test_data_file = '/home/wsx/dl.shengxian/data/mr/lstm.test.nopad'
        embedding_file = '/home/wsx/dl.shengxian/data/mr/word_rep_w2v.plpl'
        max_doc_len = 100
        vocab_size = 18766

        num_class = 2
        d_mem = d_mem
        d_word_rep = 300
        batch_size = 50
        n_valid = 1067
        n_test = 1067
        dp_rate = 0.5

        g_filler = gen_uniform_filter_setting(init_iterval)
        zero_filler = gen_zero_filter_setting()
        g_updater = gen_adadelta_setting()
    elif dataset == 'simulation':
        train_data_file = '/home/wsx/dl.shengxian/data/simulation/neg.gen.train.pad'
        test_data_file  = '/home/wsx/dl.shengxian/data/simulation/neg.gen.train.pad'
        # test_data_file  = '/home/wsx/dl.shengxian/data/simulation/lstm.train.negneglongterm'
        n_test = 300 
        embedding_file = ''
        max_doc_len = 100
        vocab_size = 2000

        num_class = 2
        d_word_rep = d_mem
        batch_size = 1
        dp_rate = 0

        g_filler = gen_uniform_filter_setting(0.003)
        zero_filler = gen_zero_filter_setting()
        g_updater = gen_adadelta_setting()
    elif dataset == 'treebank':
        assert False

    g_layer_setting = {}
    g_layer_setting['no_bias'] = True
    g_layer_setting['phrase_type'] = 2
    g_layer_setting['w_filler'] = g_filler 
    g_layer_setting['u_filler'] = g_filler
    g_layer_setting['b_filler'] = zero_filler
    g_layer_setting['w_updater'] = g_updater
    g_layer_setting['u_updater'] = g_updater
    g_layer_setting['b_updater'] = g_updater

    net = {}
    net['net_name'] = 'rnn'
    net['cross_validation'] = 10
    net['log'] = 'log.rnn.max.cv'
    net_cfg_train, net_cfg_valid, net_cfg_test = {}, {}, {}
    net['net_config'] = [net_cfg_train, net_cfg_valid, net_cfg_test]
    net_cfg_train["tag"] = "Train"
    net_cfg_train["max_iters"] = 2000
    net_cfg_train["display_interval"] = 100
    # net_cfg_train["out_nodes"] = ['loss', 'acc']
    net_cfg_train["out_nodes"] = ['acc']
    net_cfg_valid["tag"] = "Valid"
    net_cfg_valid["max_iters"] = int(n_valid/batch_size) 
    net_cfg_valid["display_interval"] = 100
    # net_cfg_valid["out_nodes"] = ['loss', 'acc']
    net_cfg_valid["out_nodes"] = ['acc']
    net_cfg_test["tag"] = "Test"
    net_cfg_test["max_iters"] = int(n_test/batch_size) 
    net_cfg_test["display_interval"] = 100
    # net_cfg_test["out_nodes"] = ['loss', 'acc']
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
    setting['batch_size'] = batch_size
    setting['data_file'] = train_data_file
    setting['max_doc_len'] = max_doc_len

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
    setting['batch_size'] = batch_size 
    setting['data_file'] = valid_data_file
    setting['max_doc_len'] = max_doc_len
 
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
    setting['batch_size'] = batch_size 
    setting['data_file'] = test_data_file
    setting['max_doc_len'] = max_doc_len

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['x']
    layer['top_nodes'] = ['word_rep_seq']
    layer['layer_name'] = 'embedding'
    layer['layer_type'] = 21
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['embedding_file'] = embedding_file
    setting['feat_size'] = d_word_rep
    setting['word_count'] = vocab_size

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['word_rep_seq']
    layer['top_nodes'] = ['rnn_seq']
    layer['layer_name'] = 'rnn'
    layer['layer_type'] = 27 
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['d_mem'] = d_mem
    setting['reverse'] = False
    # setting['input_transform'] = False
    # setting['nonlinear'] = 'rectifier'

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['rnn_seq']
    layer['top_nodes'] = ['pool_rep']
    layer['layer_name'] = 'wholePooling'
    layer['layer_type'] =  25 
    setting = {"phrase_type":2, "pool_type":"max"}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['pool_rep']
    layer['top_nodes'] = ['drop_rep']
    layer['layer_name'] = 'dropout'
    layer['layer_type'] =  13
    setting = {'phrase_type':2, 'rate':dp_rate}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['drop_rep']
    layer['top_nodes'] = ['softmax_ret']
    layer['layer_name'] = 'softmax_fullconnect'
    layer['layer_type'] = 11 
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['num_hidden'] = num_class
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

for d_mem in [100]:
    net = gen_rnn(d_mem = d_mem, init_iterval = 0.03)
    gen_conf_file(net, '../bin/rnn_mr.model')
    os.system("../bin/textnet ../bin/rnn_mr.model")
