#-*-coding:utf8-*-
import copy, os

from gen_conf_file import *
from dataset_cfg import *

def gen_match_lstm(d_mem, init, lr, dataset, l2, lstm_norm2):
    # print "ORC: left & right lstm share parameters"
    net = {}

    ds = DatasetCfg(dataset)
    g_filler    = gen_uniform_filter_setting(init)
    zero_filler = gen_zero_filter_setting()
    g_updater   = gen_adagrad_setting(lr = lr, l2 = l2, batch_size = ds.train_batch_size)
    zero_l2_updater   = gen_adagrad_setting(lr = lr, batch_size = ds.train_batch_size)

    g_layer_setting = {}
    g_layer_setting['no_bias'] = False

    g_layer_setting['w_filler'] = g_filler 
    g_layer_setting['b_filler'] = zero_filler
    g_layer_setting['w_updater'] = g_updater
    g_layer_setting['b_updater'] = g_updater

    net['net_name'] = 'match_multigran_cnn_tensor_dpool'
    net['need_reshape'] = True
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
    print "ORC: update all words"
    # setting['update_indication_file'] = ds.update_indication_file
    setting['feat_size'] = ds.d_word_rep
    setting['word_count'] = ds.vocab_size
    print "ORC: not use l2 for embedding"
    setting['w_updater'] = zero_l2_updater

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['word_rep_seq']
    layer['top_nodes'] = ['l_sentence', 'r_sentence']
    layer['layer_name'] = 'sentence_split'
    layer['layer_type'] = 20 
    setting = {}
    layer['setting'] = setting

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['l_sentence']
    layer['top_nodes'] = ['l_sentence_conv_1']
    layer['layer_name'] = 'l_conv_1'
    layer['layer_type'] = 14
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['channel_out'] = d_mem 
    setting['kernel_x'] = ds.d_word_rep
    setting['kernel_y'] = 3
    setting['pad_x'] = 0
    setting['pad_y'] = 2
    setting['no_bias'] = True
    setting['stride'] = 1
    # setting['d1_var_len'] = True 

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['r_sentence']
    layer['top_nodes'] = ['r_sentence_conv_1']
    layer['layer_name'] = 'r_conv_1'
    layer['layer_type'] = 14
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['channel_out'] = d_mem 
    setting['kernel_x'] = ds.d_word_rep 
    setting['kernel_y'] = 3
    setting['pad_x'] = 0
    setting['pad_y'] = 2
    setting['no_bias'] = True
    setting['stride'] = 1
    # setting['d1_var_len'] = True 

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['l_sentence_conv_1', 'r_sentence_conv_1']
    layer['top_nodes'] = ['cross']
    layer['layer_name'] = 'cross_layer'
    layer['layer_type'] = 22 
    setting = {}
    layer['setting'] = setting

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['cross']
    layer['top_nodes'] = ['pool1']
    layer['layer_name'] = 'maxpool1'
    layer['layer_type'] = 15 
    setting = {'kernel_x':2, 'kernel_y':2, 'stride':2}
    layer['setting'] = setting

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['pool1']
    layer['top_nodes'] = ['relu1']
    layer['layer_name'] = 'nonlinear_1'
    layer['layer_type'] = 1 
    setting = {}
    layer['setting'] = setting

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['relu1']
    layer['top_nodes'] = ['conv_2']
    layer['layer_name'] = 'r_conv_1'
    layer['layer_type'] = 14
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['channel_out'] = d_mem 
    setting['kernel_x'] = 2 
    setting['kernel_y'] = 2
    setting['pad_x'] = 0
    setting['pad_y'] = 0
    setting['no_bias'] = True
    setting['stride'] = 1

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['conv_2']
    layer['top_nodes'] = ['pool2']
    layer['layer_name'] = 'maxpool2'
    layer['layer_type'] = 15 
    setting = {'kernel_x':2, 'kernel_y':2, 'stride':2}
    layer['setting'] = setting

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['pool2']
    layer['top_nodes'] = ['relu2']
    layer['layer_name'] = 'nonlinear_2'
    layer['layer_type'] = 1 
    setting = {}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['relu2']
    layer['top_nodes'] = ['hidden_trans']
    layer['layer_name'] = 'mlp_hidden'
    layer['layer_type'] = 11
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['num_hidden'] = d_mem * 2

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['hidden_trans']
    layer['top_nodes'] = ['hidden_rep']
    layer['layer_name'] = 'hidden_nonlinear'
    layer['layer_type'] = 1
    setting = {}
    layer['setting'] = setting
     
    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['hidden_rep']
    # layer['bottom_nodes'] = ['dpool_rep']
    layer['top_nodes'] = ['softmax_prob']
    layer['layer_name'] = 'softmax_fullconnect'
    layer['layer_type'] = 11
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['num_hidden'] = ds.num_class
    # setting['no_bias'] = True
    setting['w_filler'] = zero_filler

    layer = {} 
    layers.append(layer) 
    layer['bottom_nodes'] = ['softmax_prob', 'y']
    layer['top_nodes'] = ['loss']
    layer['layer_name'] = 'softmax_activation'
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
for dataset in ['qa_balance']:
    for d_mem in [50]:
        idx = 0
        # for epoch_no in [0, 10000, 25000]:
        for epoch_no in [0]:
            for init in [0.3, 0.1, 0.03]:
                for lr in [0.3, 0.1, 0.03]:
                    # for l2 in [0.00001, 0.0001, 0.001]:
                    init_t = init
                    # t_lr = t_lr_mul * lr
                    pretrain_run_no = 0 
                    lstm_norm2 = 100000 
                    net = gen_match_lstm(d_mem=d_mem,init=init,lr=lr,dataset=dataset,l2=l2,lstm_norm2=lstm_norm2)
                    net['log'] = 'log.match.arcii.{0}.d{1}.run{2}.{3}'.format\
                                 (dataset, str(d_mem), str(run), str(idx))
                    # net["save_model"] = {"file_prefix": "./model/model."+str(idx),"save_interval": 500}
                    # net["save_activation"] = [{"tag":"Valid","file_prefix": \
                    #                            "./model/valid."+str(idx), \
                    #                            "save_interval": 500, \
                    #                            "save_nodes":["x","y","word_rep_seq","l_sentence",\
                    #                                          "r_sentence","interaction_rep", \
                    #                                          # "interaction_rep_nonlinear",\
                    #                                          "dpool_rep","softmax_prob"], \
                    #                            "save_iter_num":1}]


                    gen_conf_file(net, '/home/wsx/exp/match/{0}/arcii/run.{1}/'.format(dataset,str(run)) + \
                                       'model.match.arcii.{0}.d{1}.run{2}.{3}'.format\
                                       (dataset, str(d_mem), str(run), str(idx)))
                    idx += 1
