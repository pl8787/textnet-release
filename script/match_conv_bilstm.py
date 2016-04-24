#-*-coding:utf8-*-
import copy, os

from gen_conf_file import *
from dataset_cfg import *

def gen_conv_bilstm(d_mem, init, l2, lr, dataset, batch_size, lstm_norm2):
    is_use_mlp= True
    use_gru = True
    is_deep   = False 
    print "ORC: using MLP"

    # print "ORC: left & right lstm share parameters"
    net = {}

    ds = DatasetCfg(dataset)
    g_filler    = gen_uniform_filler_setting(init)
    zero_filler = gen_zero_filler_setting()
    t_updater   = gen_adagrad_setting(lr = t_lr, l2 = t_l2, batch_size = ds.train_batch_size)
    g_updater   = gen_adagrad_setting(lr = lr, l2 = l2, batch_size = ds.train_batch_size)
    print "ORC: word embedding learning rate rescale."
    zero_l2_updater   = gen_adagrad_setting(lr = lr*0.3, batch_size = ds.train_batch_size)
    # g_updater   = gen_sgd_setting(lr = lr, l2 = l2, batch_size = ds.train_batch_size)
    # zero_l2_updater   = gen_sgd_setting(lr = lr, batch_size = ds.train_batch_size)

    g_layer_setting = {}
    g_layer_setting['no_bias'] = False

    g_layer_setting['t_filler'] = g_filler 
    g_layer_setting['w_filler'] = g_filler 
    g_layer_setting['b_filler'] = zero_filler
    g_layer_setting['w_g_filler'] = g_filler 
    g_layer_setting['b_g_filler'] = zero_filler
    g_layer_setting['w_c_filler'] = g_filler 
    g_layer_setting['b_c_filler'] = zero_filler
    g_layer_setting['t_updater'] = t_updater
    g_layer_setting['w_updater'] = g_updater
    g_layer_setting['b_updater'] = g_updater
    g_layer_setting['w_g_updater'] = g_updater
    g_layer_setting['b_g_updater'] = g_updater
    g_layer_setting['w_c_updater'] = g_updater
    g_layer_setting['b_c_updater'] = g_updater

    net['net_name'] = 'match_conv_lstm'
    net['need_reshape'] = True
    net_cfg_train, net_cfg_valid, net_cfg_test = {}, {}, {}
    net['net_config'] = [net_cfg_train, net_cfg_valid, net_cfg_test]
    net_cfg_train["tag"] = "Train"
    net_cfg_train["max_iters"] = ds.train_max_iters * 2
    net_cfg_train["display_interval"] = ds.train_display_interval
    net_cfg_train["out_nodes"] = ['loss']
    net_cfg_valid["tag"] = "Valid"
    net_cfg_valid["max_iters"] = ds.valid_max_iters 
    net_cfg_valid["display_interval"] = ds.valid_display_interval 
    net_cfg_valid["out_nodes"] = ['P@k','MRR']
    net_cfg_test["tag"]  = "Test"
    net_cfg_test["max_iters"]  = ds.test_max_iters 
    net_cfg_test["display_interval"] = ds.test_display_interval
    net_cfg_test["out_nodes"]  = ['P@k','MRR']
    layers = []
    net['layers'] = layers

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['x', 'y']
    layer['layer_name'] = 'train_data'
    layer['layer_type'] = 79
    layer['tag'] = ['Train']
    setting = {}
    layer['setting'] = setting
    setting['batch_size'] = ds.train_batch_size
    setting['shuffle'] = True
    setting['data_file'] = ds.train_data_file
    setting['max_doc_len'] = ds.max_doc_len
    setting['min_doc_len'] = ds.min_doc_len

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['x', 'y']
    layer['layer_name'] = 'valid_data'
    layer['layer_type'] = 80 
    layer['tag'] = ['Valid']
    setting = {}
    layer['setting'] = setting
    setting['data_file'] = ds.valid_data_file
    setting['max_doc_len'] = ds.max_doc_len
    setting['min_doc_len'] = ds.min_doc_len

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['x', 'y']
    layer['layer_name'] = 'test_data'
    layer['layer_type'] = 80 
    layer['tag'] = ['Test']
    setting = {}
    layer['setting'] = setting
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
    layer['bottom_nodes'] = ['word_rep_seq']
    layer['top_nodes'] = ['r_lstm_seq']
    layer['layer_name'] = 'r_lstm'
    layer['layer_type'] = 24
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['d_mem'] = d_mem
    setting['reverse'] = True 
    setting['grad_norm2'] = lstm_norm2

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['l_lstm_seq', 'r_lstm_seq']
    layer['top_nodes'] = ['bi_lstm_seq']
    layer['layer_name'] = 'concat'
    layer['layer_type'] = 18
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['concat_dim_index'] = 1
    setting['bottom_node_num'] = 2

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['bi_lstm_seq']
    layer['top_nodes'] = ['conv_seq']
    layer['layer_name'] = 'conv'
    layer['layer_type'] = 14
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['channel_out'] = d_mem*2
    setting['kernel_y'] = 1
    setting['pad_y'] = setting['kernel_y'] - 1
    setting['kernel_x'] = d_mem 
    setting['d1_var_len'] = True

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['conv_seq']
    layer['top_nodes'] = ['conv_activation']
    layer['layer_name'] = 'nonlinear'
    layer['layer_type'] = 1 
    setting = {}
    layer['setting'] = setting
    
    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['conv_activation']
    layer['top_nodes'] = ['conv_ret_trans']
    layer['layer_name'] = 'conv_ret_transform'
    layer['layer_type'] =  32 
    setting = {}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['conv_ret_trans']
    layer['top_nodes'] = ['pool_rep']
    layer['layer_name'] = 'wholePooling'
    layer['layer_type'] =  25
    setting = {"pool_type":"max"}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['pool_rep']
    layer['top_nodes'] = ['drop_rep']
    layer['layer_name'] = 'dropout'
    layer['layer_type'] =  13
    ds.dp_rate = 0.
    print "ORC, dp_rate:", ds.dp_rate
    setting = {'rate':ds.dp_rate}
    layer['setting'] = setting

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

run = 2 
lr = 0.
for dataset in ['trec']:
    for d_mem in [50]:
        idx = 0
        for init in [0.5, 0.3]:
            for l2 in [0, 0.000001, 0.000003, 0.00001]:# , 0.00001, 0.0001, 0.001]:
                for lstm_norm2 in [2, 1, 0.5]:
                    for batch_size in [10, 30, 50]:
                        net = gen_conv_bilstm(d_mem=d_mem, init=init, lr=lr, dataset=dataset, \
                                              l2=l2, batch_size=batch_size, lstm_norm2=lstm_norm2)
                        net['log'] = 'log.conv_bilstm.max.{0}.d{1}.run{2}.{3}'.\
                                      format(dataset, str(d_mem), str(run),str(idx))
                        gen_conf_file(net, '/home/wsx/exp/ccir2015/{0}/conv_bilstm/run.2/model.conv_bilstm.max.{1}.d{2}.run{3}.{4}'.\
                                      format(dataset, dataset, str(d_mem), str(run), str(idx)))
                        idx += 1
