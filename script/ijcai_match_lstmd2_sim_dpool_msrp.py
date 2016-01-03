#-*-coding:utf8-*-
import copy, os

from gen_conf_file import *
from dataset_cfg import *

f_gate_bias = 0.
o_gate_bias = 0.
i_gate_bias = 0.

def gen_match_lstm(d_mem, init, lr, dataset, l2, lstm_norm2):
    print "Without using MLP."
    is_use_mlp = False 
    net = {}

    ds = DatasetCfg(dataset)
    g_filler    = gen_uniform_filler_setting(init)
    zero_filler = gen_zero_filler_setting()
    g_updater   = gen_adagrad_setting(lr = lr, l2 = l2, batch_size = ds.train_batch_size)
    zero_l2_updater   = gen_adagrad_setting(lr = lr*0.1, batch_size = ds.train_batch_size)

    g_layer_setting = {}
    g_layer_setting['no_bias'] = False
    g_layer_setting['w_filler'] = g_filler 
    g_layer_setting['b_filler'] = zero_filler
    g_layer_setting['w_updater'] = g_updater
    g_layer_setting['b_updater'] = g_updater


    net['net_name'] = 'match_bilstm_sim_dpool'
    net['need_reshape'] = False
    net_cfg_train, net_cfg_valid, net_cfg_test = {}, {}, {}
    net['net_config'] = [net_cfg_train, net_cfg_valid, net_cfg_test]
    net_cfg_train["tag"] = "Train"
    net_cfg_train["max_iters"] = ds.train_max_iters * 4
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
    setting['batch_size'] = ds.train_batch_size
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
    setting['batch_size'] = ds.train_batch_size
    setting['data_file'] = ds.test_data_file
    setting['max_doc_len'] = ds.max_doc_len
    setting['min_doc_len'] = ds.min_doc_len

    # layer = {}
    # layers.append(layer) 
    # layer['bottom_nodes'] = ['x']
    # layer['top_nodes'] = ['word_rep_seq']
    # layer['layer_name'] = 'embedding'
    # layer['layer_type'] = 21
    # setting = copy.deepcopy(g_layer_setting)
    # layer['setting'] = setting
    # setting['embedding_file'] = ds.embedding_file
    # # print "ORC: update all words"
    # setting['update_indication_file'] = ds.update_indication_file
    # setting['feat_size'] = ds.d_word_rep
    # setting['word_count'] = ds.vocab_size
    # print "ORC: not use l2 for embedding"
    # setting['w_updater'] = zero_l2_updater

    layer = {}
    layers.append(layer) 
    # layer['bottom_nodes'] = ['word_rep_seq']
    layer['bottom_nodes'] = ['x']
    layer['top_nodes'] = ['l_sentence', 'r_sentence']
    layer['layer_name'] = 'sentence_split'
    layer['layer_type'] = 20 
    setting = {}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['l_sentence', 'r_sentence']
    layer['top_nodes'] = ['dot_similarity']
    layer['layer_name'] = 'match'
    layer['layer_type'] = 23 
    print "ORC: use DOT operation for similarity"
    layer['setting'] = {'op':'xor'}

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['dot_similarity']
    layer['top_nodes'] = ['swap_interaction']
    layer['layer_name'] = 'swap_4_lstm_d2_1'
    layer['layer_type'] = 42 
    setting = {'axis1':1, 'axis2':2}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['swap_interaction']
    layer['top_nodes'] = ['lstm_d2_input']
    layer['layer_name'] = 'swap_4_lstm_d2_2'
    layer['layer_type'] = 42
    setting = {'axis1':2, 'axis2':3}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['lstm_d2_input']
    layer['top_nodes'] = ['match_matrix_lt2br'] # left top to bottom right
    layer['layer_name'] = 'lstm_d2_lt2br'
    layer['layer_type'] = 10005
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    print "ORC: LSTM dim:", d_mem 
    setting['d_mem'] = d_mem 
    setting['reverse'] = False
    setting['f_gate_bias_init'] = f_gate_bias
    setting['o_gate_bias_init'] = 0.0
    setting['i_gate_bias_init'] = i_gate_bias 

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['lstm_d2_input']
    layer['top_nodes'] = ['match_matrix_br2lt']
    layer['layer_name'] = 'lstm_d2_br2lt'
    layer['layer_type'] = 10005
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    print "ORC: LSTM dim:", d_mem
    setting['d_mem'] = d_mem 
    setting['reverse'] = True
    setting['f_gate_bias_init'] = f_gate_bias
    setting['o_gate_bias_init'] = 0.0
    setting['i_gate_bias_init'] = i_gate_bias 

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['match_matrix_lt2br','match_matrix_br2lt']
    layer['top_nodes'] = ['match_matrix']
    layer['layer_name'] = 'concat'
    layer['layer_type'] = 18
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['bottom_node_num'] = 2
    setting['concat_dim_index'] = 3

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['match_matrix']
    layer['top_nodes'] = ['match_matrix_swap']
    layer['layer_name'] = 'swap_4_dpool_1'
    layer['layer_type'] = 42 
    setting = {'axis1':3, 'axis2':2}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['match_matrix_swap']
    layer['top_nodes'] = ['match_matrix_output']
    layer['layer_name'] = 'swap_4_dpool_2'
    layer['layer_type'] = 42
    setting = {'axis1':2, 'axis2':1}
    layer['setting'] = setting

    # print "USE NON LINEAR ON DOT"
    # layer = {}
    # layers.append(layer) 
    # layer['bottom_nodes'] = ['dot_similarity']
    # layer['top_nodes'] = ['dot_similarity_nonlinear']
    # layer['layer_name'] = 'dot_nonlinear'
    # layer['layer_type'] = 3 
    # setting = {}
    # layer['setting'] = setting
         
    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['match_matrix_output', 'l_sentence', 'r_sentence']
    layer['top_nodes'] = ['dpool_rep']
    layer['layer_name'] = 'dynamic_pooling'
    layer['layer_type'] = 43
    layer['setting'] = {'row':5, 'col':5}

    if is_use_mlp:
        layer = {}
        layers.append(layer) 
        layer['bottom_nodes'] = ['dpool_rep']
        layer['top_nodes'] = ['hidden_trans']
        layer['layer_name'] = 'mlp_hidden'
        layer['layer_type'] = 11 
        setting = copy.deepcopy(g_layer_setting)
        layer['setting'] = setting
        setting['num_hidden'] = d_mem * 4

        layer = {}
        layers.append(layer) 
        layer['bottom_nodes'] = ['hidden_trans']
        layer['top_nodes'] = ['hidden_rep']
        layer['layer_name'] = 'hidden_nonlinear'
        layer['layer_type'] = 3 
        setting = {}
        layer['setting'] = setting
        
    layer = {}
    layers.append(layer) 
    if is_use_mlp:
        layer['bottom_nodes'] = ['hidden_rep']
    else:
        layer['bottom_nodes'] = ['dpool_rep']
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

run = 4
l2 = 0.
for dataset in ['msrp']:
    for d_mem in [5]:
        idx = 0
        for init in [0.5, 0.1, 0.03]:
            for lr in [0.5, 0.3, 0.1]:
                # for l2 in [0.00001, 0.0001]:
                for l2 in [0]:
                    lstm_norm2 = 10000 
                    net = gen_match_lstm(d_mem=d_mem,init=init,lr=lr,dataset=dataset,l2=l2,lstm_norm2=lstm_norm2)
                    net['log'] = 'log.match.lstmd2_sim_dpool.{0}.d{1}.run{2}.{3}'.format\
                                 (dataset, str(d_mem), str(run), str(idx))
                    gen_conf_file(net, '/home/wsx/exp/match/{0}/lstmd2/run.{1}/'.format(dataset,str(run)) + \
                                       'model.match.lstmd2_sim_dpool.{0}.d{1}.run{2}.{3}'.format\
                                       (dataset, str(d_mem), str(run), str(idx)))
                    idx += 1
