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

    g_layer_setting['t_filler'] = g_filler 
    g_layer_setting['w_filler'] = g_filler 
    g_layer_setting['b_filler'] = zero_filler
    g_layer_setting['t_updater'] = g_updater
    g_layer_setting['w_updater'] = g_updater
    g_layer_setting['b_updater'] = g_updater

    net['net_name'] = 'cnn_tensor'
    net['need_reshape'] = True 
    net_cfg_train, net_cfg_valid, net_cfg_test = {}, {}, {}
    net['net_config'] = [net_cfg_train, net_cfg_valid, net_cfg_test]
    net_cfg_train["tag"] = "Train"
    net_cfg_train["max_iters"] = ds.train_max_iters 
    net_cfg_train["display_interval"] = ds.train_display_interval
    net_cfg_train["out_nodes"] = ['loss']
    net_cfg_valid["tag"] = "Valid"
    net_cfg_valid["max_iters"] = ds.valid_max_iters 
    net_cfg_valid["display_interval"] = ds.valid_display_interval 
    net_cfg_valid["out_nodes"] = ['P@k','MRR']
    net_cfg_test["tag"] = "Test"
    net_cfg_test["max_iters"]  = ds.test_max_iters 
    net_cfg_test["display_interval"] = ds.test_display_interval
    net_cfg_test["out_nodes"]  = ['P@k', 'MRR']
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
    setting['batch_size'] = ds.valid_batch_size
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
    setting['d1_var_len'] = True 

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['l_sentence_conv_1']
    layer['top_nodes'] = ['l_sentence_conv_nonlinear_1']
    layer['layer_name'] = 'l_conv_nonlinear_1'
    layer['layer_type'] = 3 
    setting = {}
    layer['setting'] = setting

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['l_sentence_conv_nonlinear_1']
    layer['top_nodes'] = ['l_sentence_swap_1']
    layer['layer_name'] = 'l_swap_1'
    layer['layer_type'] = 42
    setting = {'pass_len':True, 'pass_len_dim':1, 'axis1':1, 'axis2':3}
    layer['setting'] = setting

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['l_sentence_swap_1', 'l_sentence']
    layer['top_nodes'] = ['l_sentence_pool_1']
    layer['layer_name'] = 'l_pool_1'
    layer['layer_type'] = 10001 
    setting = {}
    layer['setting'] = setting
    setting['L'] = 2
    setting['l'] = 1
    setting['max_sentence_length'] = ds.max_doc_len
    setting['min_rep_length'] = 4

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['l_sentence_pool_1']
    layer['top_nodes'] = ['l_sentence_conv_2']
    layer['layer_name'] = 'l_conv_2'
    layer['layer_type'] = 14
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['channel_out'] = d_mem 
    setting['kernel_x'] = d_mem 
    setting['kernel_y'] = 3
    setting['pad_x'] = 0
    setting['pad_y'] = 2
    setting['no_bias'] = True
    setting['stride'] = 1
    setting['d1_var_len'] = True 

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['l_sentence_conv_2']
    layer['top_nodes'] = ['l_sentence_conv_nonlinear_2']
    layer['layer_name'] = 'l_conv_nonlinear_2'
    layer['layer_type'] = 3 
    setting = {}
    layer['setting'] = setting

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['l_sentence_conv_nonlinear_2']
    layer['top_nodes'] = ['l_sentence_swap_2']
    layer['layer_name'] = 'l_swap_2'
    layer['layer_type'] = 42
    setting = {'pass_len':True, 'pass_len_dim':1, 'axis1':1, 'axis2':3}
    layer['setting'] = setting

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['l_sentence_swap_2', 'l_sentence']
    layer['top_nodes'] = ['l_sentence_pool_2']
    layer['layer_name'] = 'l_pool_2'
    layer['layer_type'] = 10001 
    setting = {}
    layer['setting'] = setting
    setting['L'] = 2
    setting['l'] = 2
    setting['max_sentence_length'] = ds.max_doc_len
    setting['min_rep_length'] = 4

    # layer = {}
    # layers.append(layer)
    # layer['bottom_nodes'] = ['l_sentence_pool_2']
    # layer['top_nodes'] = ['l_sentence_conv_3']
    # layer['layer_name'] = 'l_conv_3'
    # layer['layer_type'] = 14
    # setting = copy.deepcopy(g_layer_setting)
    # layer['setting'] = setting
    # setting['channel_out'] = d_mem 
    # setting['kernel_x'] = d_mem 
    # setting['kernel_y'] = 3
    # setting['pad_x'] = 0
    # setting['pad_y'] = 2
    # setting['no_bias'] = True
    # setting['stride'] = 1
    # setting['d1_var_len'] = True 

    # layer = {}
    # layers.append(layer) 
    # layer['bottom_nodes'] = ['l_sentence_conv_3']
    # layer['top_nodes'] = ['l_sentence_conv_nonlinear_3']
    # layer['layer_name'] = 'l_conv_nonlinear_3'
    # layer['layer_type'] = 3 
    # setting = {}
    # layer['setting'] = setting

    # layer = {}
    # layers.append(layer)
    # layer['bottom_nodes'] = ['l_sentence_conv_nonlinear_3']
    # layer['top_nodes'] = ['l_sentence_swap_3']
    # layer['layer_name'] = 'l_swap_3'
    # layer['layer_type'] = 42
    # setting = {'pass_len':True, 'pass_len_dim':1, 'axis1':1, 'axis2':3}
    # layer['setting'] = setting

    # layer = {}
    # layers.append(layer)
    # layer['bottom_nodes'] = ['l_sentence_swap_3', 'l_sentence']
    # layer['top_nodes'] = ['l_sentence_pool_3']
    # layer['layer_name'] = 'l_pool_3'
    # layer['layer_type'] = 10001 
    # setting = {}
    # layer['setting'] = setting
    # setting['L'] = 3
    # setting['l'] = 3
    # setting['max_sentence_length'] = ds.max_doc_len
    # setting['min_rep_length'] = 4

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
    setting['d1_var_len'] = True 

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['r_sentence_conv_1']
    layer['top_nodes'] = ['r_sentence_conv_nonlinear_1']
    layer['layer_name'] = 'r_conv_nonlinear_1'
    layer['layer_type'] = 3 
    setting = {}
    layer['setting'] = setting

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['r_sentence_conv_nonlinear_1']
    layer['top_nodes'] = ['r_sentence_swap_1']
    layer['layer_name'] = 'r_swap_1'
    layer['layer_type'] = 42
    setting = {'pass_len':True, 'pass_len_dim':1, 'axis1':1, 'axis2':3}
    layer['setting'] = setting

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['r_sentence_swap_1', 'r_sentence']
    layer['top_nodes'] = ['r_sentence_pool_1']
    layer['layer_name'] = 'r_pool_1'
    layer['layer_type'] = 10001 
    setting = {}
    layer['setting'] = setting
    setting['L'] = 2
    setting['l'] = 1
    setting['max_sentence_length'] = ds.max_doc_len
    setting['min_rep_length'] = 4

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['r_sentence_pool_1']
    layer['top_nodes'] = ['r_sentence_conv_2']
    layer['layer_name'] = 'r_conv_2'
    layer['layer_type'] = 14
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['channel_out'] = d_mem 
    setting['kernel_x'] = d_mem 
    setting['kernel_y'] = 3
    setting['pad_x'] = 0
    setting['pad_y'] = 2
    setting['no_bias'] = True
    setting['stride'] = 1
    setting['d1_var_len'] = True 

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['r_sentence_conv_2']
    layer['top_nodes'] = ['r_sentence_conv_nonlinear_2']
    layer['layer_name'] = 'r_conv_nonlinear_2'
    layer['layer_type'] = 3 
    setting = {}
    layer['setting'] = setting

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['r_sentence_conv_nonlinear_2']
    layer['top_nodes'] = ['r_sentence_swap_2']
    layer['layer_name'] = 'r_swap_2'
    layer['layer_type'] = 42
    setting = {'pass_len':True, 'pass_len_dim':1, 'axis1':1, 'axis2':3}
    layer['setting'] = setting

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['r_sentence_swap_2', 'r_sentence']
    layer['top_nodes'] = ['r_sentence_pool_2']
    layer['layer_name'] = 'r_pool_2'
    layer['layer_type'] = 10001 
    setting = {}
    layer['setting'] = setting
    setting['L'] = 2
    setting['l'] = 2
    setting['max_sentence_length'] = ds.max_doc_len
    setting['min_rep_length'] = 4

    # layer = {}
    # layers.append(layer)
    # layer['bottom_nodes'] = ['r_sentence_pool_2']
    # layer['top_nodes'] = ['r_sentence_conv_3']
    # layer['layer_name'] = 'r_conv_3'
    # layer['layer_type'] = 14
    # setting = copy.deepcopy(g_layer_setting)
    # layer['setting'] = setting
    # setting['channel_out'] = d_mem 
    # setting['kernel_x'] = d_mem 
    # setting['kernel_y'] = 3
    # setting['pad_x'] = 0
    # setting['pad_y'] = 2
    # setting['no_bias'] = True
    # setting['stride'] = 1
    # setting['d1_var_len'] = True 

    # layer = {}
    # layers.append(layer) 
    # layer['bottom_nodes'] = ['r_sentence_conv_3']
    # layer['top_nodes'] = ['r_sentence_conv_nonlinear_3']
    # layer['layer_name'] = 'r_conv_nonlinear_3'
    # layer['layer_type'] = 3 
    # setting = {}
    # layer['setting'] = setting

    # layer = {}
    # layers.append(layer)
    # layer['bottom_nodes'] = ['r_sentence_conv_nonlinear_3']
    # layer['top_nodes'] = ['r_sentence_swap_3']
    # layer['layer_name'] = 'r_swap_3'
    # layer['layer_type'] = 42
    # setting = {'pass_len':True, 'pass_len_dim':1, 'axis1':1, 'axis2':3}
    # layer['setting'] = setting

    # layer = {}
    # layers.append(layer)
    # layer['bottom_nodes'] = ['r_sentence_swap_3', 'r_sentence']
    # layer['top_nodes'] = ['r_sentence_pool_3']
    # layer['layer_name'] = 'r_pool_3'
    # layer['layer_type'] = 10001 
    # setting = {}
    # layer['setting'] = setting
    # setting['L'] = 3
    # setting['l'] = 3
    # setting['max_sentence_length'] = ds.max_doc_len
    # setting['min_rep_length'] = 4

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['l_sentence_pool_2', 'r_sentence_pool_2']
    layer['top_nodes'] = ['tensor_trans']
    layer['layer_name'] = 'match_tensor'
    layer['layer_type'] = 1001
    # print "ORC: use COS operation for similarity"
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['d_hidden'] = 1
    # setting['d_factor'] = 3*d_mem 
    # setting['t_l2'] = t_l2
    # setting['is_init_as_I'] = False
    # setting['is_init_as_I'] = True
    setting['is_use_linear'] = False
    setting['is_var_len'] = False
    # setting['is_update_tensor'] = False
    # setting['is_update_tensor'] = True
    # setting['t_updater'] = t_updater
    # setting['w_updater'] = t_updater
    setting['t_filler'] = gen_uniform_filter_setting(init_t)
    setting['w_filler'] = gen_uniform_filter_setting(init_t)

    # layer = {}
    # layers.append(layer)
    # layer['bottom_nodes'] = ['l_sentence_pool_2', 'r_sentence_pool_2']
    # layer['top_nodes'] = ['bi_sentence_rep']
    # layer['layer_name'] = 'concat'
    # layer['layer_type'] = 18
    # setting = {}
    # layer['setting'] = setting
    # setting['bottom_node_num']  = 2
    # setting['concat_dim_index'] = 3
    # setting['is_concat_by_length'] = False 

    # layer = {}
    # layers.append(layer) 
    # layer['bottom_nodes'] = ['bi_sentence_rep']
    # layer['top_nodes'] = ['tensor_trans']
    # layer['layer_name'] = 'tensor_layer'
    # layer['layer_type'] = 30
    # setting = copy.deepcopy(g_layer_setting)
    # layer['setting'] = setting
    # setting['num_hidden'] = 1

    # layer = {}
    # layers.append(layer) 
    # layer['bottom_nodes'] = ['tensor_trans']
    # layer['top_nodes'] = ['hidden_rep']
    # layer['layer_name'] = 'hidden_nonlinear'
    # layer['layer_type'] = 0
    # setting = {}
    # layer['setting'] = setting
     
    # layer = {}
    # layers.append(layer) 
    # layer['bottom_nodes'] = ['hidden_rep']
    # layer['top_nodes'] = ['hidden_drop_rep']
    # layer['layer_name'] = 'dropout'
    # layer['layer_type'] =  13
    # ds.dp_rate = 0.
    # print "ORC, dp rate:", ds.dp_rate
    # setting = {'rate':ds.dp_rate}
    # layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    # layer['bottom_nodes'] = ['hidden_rep']
    layer['bottom_nodes'] = ['tensor_trans']
    # layer['bottom_nodes'] = ['dpool_rep']
    layer['top_nodes'] = ['softmax_prob']
    layer['layer_name'] = 'softmax_fullconnect'
    layer['layer_type'] = 11
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['num_hidden'] = 1 # ds.num_class
    # setting['no_bias'] = True
    setting['w_filler'] = zero_filler

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['softmax_prob', 'y']
    layer['top_nodes'] = ['loss']
    # layer['layer_name'] = 'softmax_activation'
    layer['layer_name'] = 'pair_hinge'
    layer['layer_type'] = 55
    layer['tag'] = ['Train'] 
    setting = {}
    layer['setting'] = setting
    setting['delta'] = 1.

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['softmax_prob', 'y']
    layer['top_nodes'] = ['P@k']
    layer['layer_name'] = 'P@k_layer'
    layer['layer_type'] = 61 
    layer['tag'] = ['Valid', 'Test'] 
    setting = {'k':1, 'col':0, 'method':'P@k'}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['softmax_prob', 'y']
    layer['top_nodes'] = ['MRR']
    layer['layer_name'] = 'MRR_layer'
    layer['layer_type'] = 61 
    layer['tag'] = ['Valid', 'Test'] 
    setting = {'k':1, 'col':0, 'method':'MRR'}
    layer['setting'] = setting

    return net

run = 3
l2 = 0.
# for dataset in ['paper']:
for dataset in ['qa_50']:
    for d_mem in [20]:
        idx = 0
        # for epoch_no in [0, 10000, 25000]:
        for epoch_no in [0]:
            for init in [0.1, 0.03, 0.01]:
                for lr in [0.3, 0.1, 0.03]:
                    # for l2 in [0.00001, 0.0001, 0.001]:
                    init_t = init
                    # t_lr = t_lr_mul * lr
                    pretrain_run_no = 0 
                    lstm_norm2 = 100000 
                    net = gen_match_lstm(d_mem=d_mem,init=init,lr=lr,dataset=dataset,l2=l2,lstm_norm2=lstm_norm2)
                    net['log'] = 'log.match.ctnn.{0}.d{1}.run{2}.{3}'.format\
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


                    gen_conf_file(net, '/home/wsx/exp/match/{0}/ctnn/run.{1}/'.format(dataset,str(run)) + \
                                       'model.match.ctnn.{0}.d{1}.run{2}.{3}'.format\
                                       (dataset, str(d_mem), str(run), str(idx)))
                    idx += 1
