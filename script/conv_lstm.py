#-*-coding:utf8-*-
import copy, os

from gen_conf_file import *


def gen_conv_lstm(d_mem_):
    dataset = 'simulation'
    if dataset == 'mr':
        train_data_file = '/home/wsx/dl.shengxian/data/mr/lstm.train.nopad'
        test_data_file = '/home/wsx/dl.shengxian/data/mr/lstm.test.pad'
        embedding_file = '/home/wsx/dl.shengxian/data/mr/word_rep_w2v.plpl'
        max_doc_len = 100
        vocab_size = 18766

        num_class = 2
        d_mem = 100
        d_word_rep = 300 
        batch_size = 50
        n_test = 1067

        g_filler = gen_uniform_filter_setting(0.3)
        zero_filler = gen_zero_filter_setting()
        g_updater = gen_adadelta_setting()
    elif dataset == 'simulation':
        train_data_file = '/home/wsx/dl.shengxian/data/simulation/neg.gen.train'
        # test_data_file  = '/home/wsx/dl.shengxian/data/simulation/neg.gen.test'
        test_data_file  = '/home/wsx/dl.shengxian/data/simulation/neg.gen.train'
        # n_test = 200 
        n_test = 300 
        # test_data_file  = '/home/wsx/dl.shengxian/data/simulation/lstm.train.negneglongterm'
        # n_test = 200 
        embedding_file = ''
        max_doc_len = 100
        vocab_size = 2000

        num_class = 2
        d_mem = d_mem_
        d_word_rep = d_mem_
        batch_size = 1

        g_filler = gen_uniform_filter_setting(0.3)
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
    net['net_name'] = 'conv_lstm'
    net['max_iters'] = 10000
    net['max_test_iters'] = int(n_test/batch_size)
    net['display_interval'] = 1
    net['test_interval'] = 122 
    net['train_out'] = ['loss', 'acc']
    net['test_out']  = ['loss', 'acc']

    layers = []
    net['layers'] = layers

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['y', 'x']
    layer['layer_name'] = 'train_data'
    layer['layer_type'] = 72
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
    layer['layer_name'] = 'test_data'
    layer['layer_type'] = 72
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
    layer['top_nodes'] = ['word_low_rep_seq']
    layer['layer_name'] = 'word_dim_reduction'
    layer['layer_type'] = 28
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['num_hidden'] = d_mem

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
    layer['bottom_nodes'] = ['word_rep_seq']
    layer['top_nodes'] = ['r_lstm_seq']
    layer['layer_name'] = 'r_lstm'
    layer['layer_type'] = 24
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['d_mem'] = d_mem
    setting['reverse'] = True 

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['l_lstm_seq', 'word_low_rep_seq', 'r_lstm_seq']
    layer['top_nodes'] = ['conv_lstm_seq']
    layer['layer_name'] = 'conv_lstm'
    layer['layer_type'] = 26
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['num_hidden'] = d_mem

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['conv_lstm_seq']
    layer['top_nodes'] = ['conv_lstm_activation']
    layer['layer_name'] = 'nonlinear'
    layer['layer_type'] = 1 
    setting = {"phrase_type":2}
    layer['setting'] = setting
        
    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['conv_lstm_activation']
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
    setting = {'phrase_type':2, 'rate':0.5}
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


    gen_conf_file(net, '../bin/conv_lstm_simulation.model')

    return

for d_mem in [3, 5, 10, 20]:
    gen_conv_lstm(d_mem)
    os.system("../bin/textnet ../bin/conv_lstm_simulation.model > ../bin/simulation/neg.gen.train.{0}".format(d_mem))
