#-*-coding:utf8-*-
import copy, os

from gen_conf_file import *
from dataset_cfg import *

t_l2 = 0.
t_lr = 0.
init_t = 0.0

def gen_match_lstm(d_mem, init, lr, dataset, l2, lstm_norm2, is_pretrain, pretrain_run_no, model_no, epoch_no):
    is_use_mlp = True
    is_deep_lstm = False
    # print "ORC: left & right lstm share parameters"
    net = {}

    ds = DatasetCfg(dataset)
    g_filler    = gen_uniform_filter_setting(init)
    zero_filler = gen_zero_filter_setting()
    t_updater   = gen_adagrad_setting(lr = t_lr, l2 = l2, batch_size = ds.train_batch_size)
    g_updater   = gen_adagrad_setting(lr = lr, l2 = l2, batch_size = ds.train_batch_size)
    zero_l2_updater   = gen_adagrad_setting(lr = lr, batch_size = ds.train_batch_size)
    # g_updater   = gen_sgd_setting(lr = lr, l2 = l2, batch_size = ds.train_batch_size)
    # zero_l2_updater   = gen_sgd_setting(lr = lr, batch_size = ds.train_batch_size)

    g_layer_setting = {}
    g_layer_setting['no_bias'] = False
    g_layer_setting['t_filler'] = g_filler 
    # g_layer_setting['t_updater'] = zero_l2_updater
    g_layer_setting['t_updater'] = t_updater
    # g_layer_setting['w_updater'] = zero_l2_updater
    # g_layer_setting['u_updater'] = zero_l2_updater
    # g_layer_setting['b_updater'] = zero_l2_updater

    g_layer_setting['w_filler'] = g_filler 
    g_layer_setting['u_filler'] = g_filler
    g_layer_setting['b_filler'] = zero_filler
    g_layer_setting['w_updater'] = g_updater
    g_layer_setting['u_updater'] = g_updater
    g_layer_setting['b_updater'] = g_updater
    g_layer_setting['w_g_filler'] = g_filler 
    g_layer_setting['u_g_filler'] = g_filler
    g_layer_setting['b_g_filler'] = zero_filler
    g_layer_setting['w_c_filler'] = g_filler 
    g_layer_setting['u_c_filler'] = g_filler
    g_layer_setting['b_c_filler'] = zero_filler
    g_layer_setting['w_g_updater'] = g_updater
    g_layer_setting['u_g_updater'] = g_updater
    g_layer_setting['b_g_updater'] = g_updater
    g_layer_setting['w_c_updater'] = g_updater
    g_layer_setting['u_c_updater'] = g_updater
    g_layer_setting['b_c_updater'] = g_updater

    net['net_name'] = 'match_bilstm_tensor_dpool'
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
    layer['top_nodes'] = ['l_lstm_seq']
    layer['layer_name'] = 'l_lstm'
    layer['layer_type'] = 24
    # layer['layer_type'] = 1006 # gru
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['d_mem'] = d_mem
    setting['grad_norm2'] = lstm_norm2
    setting['reverse'] = False
    setting['grad_cut_off'] = 10000
    setting['max_norm2'] = 10000
    # setting['f_gate_bias_init'] = 0.5
    setting['f_gate_bias_init'] = 0.
    setting['o_gate_bias_init'] = 0.

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['word_rep_seq']
    layer['top_nodes'] = ['r_lstm_seq']
    layer['layer_name'] = 'r_lstm'
    layer['layer_type'] = 24
    # layer['layer_type'] = 1006 # gru
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['d_mem'] = d_mem
    setting['grad_norm2'] = lstm_norm2
    setting['max_norm2'] = 10000
    setting['grad_cut_off'] = 10000
    setting['reverse'] = True 
    # setting['f_gate_bias_init'] = 0.5
    setting['f_gate_bias_init'] = 0.
    setting['o_gate_bias_init'] = 0.

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['word_rep_seq','l_lstm_seq', 'r_lstm_seq']
    layer['top_nodes'] = ['bi_lstm_seq']
    layer['layer_name'] = 'concat'
    layer['layer_type'] = 18
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['bottom_node_num'] = 3
    setting['concat_dim_index'] = 3

    if is_deep_lstm:
        layer = {}
        layers.append(layer) 
        layer['bottom_nodes'] = ['bi_lstm_seq']
        layer['top_nodes'] = ['l_lstm_seq_1']
        layer['layer_name'] = 'l_lstm_1'
        # layer['layer_type'] = 24
        layer['layer_type'] = 1006 # gru
        setting = copy.deepcopy(g_layer_setting)
        layer['setting'] = setting
        setting['d_mem'] = d_mem
        setting['grad_norm2'] = lstm_norm2
        setting['reverse'] = False
        setting['grad_cut_off'] = 10000
        setting['max_norm2'] = 10000
        # setting['f_gate_bias_init'] = 0.5
        setting['f_gate_bias_init'] = 0.
        setting['o_gate_bias_init'] = 0.

        layer = {}
        layers.append(layer) 
        layer['bottom_nodes'] = ['bi_lstm_seq']
        layer['top_nodes'] = ['r_lstm_seq_1']
        layer['layer_name'] = 'r_lstm_1'
        # layer['layer_type'] = 24
        layer['layer_type'] = 1006 # gru
        setting = copy.deepcopy(g_layer_setting)
        layer['setting'] = setting
        setting['d_mem'] = d_mem
        setting['grad_norm2'] = lstm_norm2
        setting['max_norm2'] = 10000
        setting['grad_cut_off'] = 10000
        setting['reverse'] = True 
        # setting['f_gate_bias_init'] = 0.5
        setting['f_gate_bias_init'] = 0.
        setting['o_gate_bias_init'] = 0.

        layer = {}
        layers.append(layer) 
        layer['bottom_nodes'] = ['word_rep_seq', 'bi_lstm_seq', 'l_lstm_seq_1', 'r_lstm_seq_1']
        layer['top_nodes'] = ['bi_lstm_seq_1']
        layer['layer_name'] = 'concat'
        layer['layer_type'] = 18
        setting = copy.deepcopy(g_layer_setting)
        layer['setting'] = setting
        setting['bottom_node_num'] = 4
        setting['concat_dim_index'] = 3

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['bi_lstm_seq']
    layer['top_nodes'] = ['l_sentence', 'r_sentence']
    layer['layer_name'] = 'sentence_split'
    layer['layer_type'] = 20 
    setting = {}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['l_sentence', 'r_sentence']
    layer['top_nodes'] = ['interaction_rep']
    layer['layer_name'] = 'match_tensor'
    layer['layer_type'] = 1003
    # print "ORC: use COS operation for similarity"
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['d_hidden'] = 5
    setting['d_factor'] = 3*d_mem 
    setting['t_l2'] = t_l2
    setting['is_init_as_I'] = False
    # setting['is_init_as_I'] = True
    setting['is_use_linear'] = True
    # setting['is_update_tensor'] = False
    setting['is_update_tensor'] = True
    setting['t_updater'] = t_updater
    setting['w_updater'] = t_updater
    setting['t_filler'] = gen_uniform_filter_setting(init_t)
    setting['w_filler'] = gen_uniform_filter_setting(init_t)

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['interaction_rep']
    layer['top_nodes'] = ['interaction_rep_nonlinear']
    layer['layer_name'] = 'tensor_nonlinear'
    layer['layer_type'] = 1 
    setting = {}
    layer['setting'] = setting
    
    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['interaction_rep_nonlinear', 'l_sentence', 'r_sentence']
    # layer['bottom_nodes'] = ['interaction_rep', 'l_sentence', 'r_sentence']
    layer['top_nodes'] = ['dpool_rep']
    layer['layer_name'] = 'dynamic_pooling'
    layer['layer_type'] = 43
    layer['setting'] = {'row':5, 'col':5, 'interval':1}

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
        layer['layer_type'] = 1 
        setting = {}
        layer['setting'] = setting
        
        layer = {}
        layers.append(layer) 
        layer['bottom_nodes'] = ['hidden_rep']
        layer['top_nodes'] = ['hidden_drop_rep']
        layer['layer_name'] = 'dropout'
        layer['layer_type'] =  13
        ds.dp_rate = 0.
        print "ORC, dp rate:", ds.dp_rate
        setting = {'rate':ds.dp_rate}
        layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    if is_use_mlp:
        layer['bottom_nodes'] = ['hidden_drop_rep']
    else:
        layer['bottom_nodes'] = ['dpool_rep']
    layer['top_nodes'] = ['softmax_prob']
    layer['layer_name'] = 'softmax_fullconnect'
    layer['layer_type'] = 11
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['num_hidden'] = 1
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

run = 2
l2 = 0.
# for dataset in ['paper']:
# for dataset in ['qa_balance']:
for dataset in ['qa_50']:
    for d_mem in [50]:
        idx = 0
        # for model_no in [0,1,2,3,4,5,6,7,8]:
        #     for epoch_no in [20000, 40000, 80000]:
        for model_no in [2]:
            # for epoch_no in [0, 10000, 25000]:
            for epoch_no in [0]:
                # for init in [0.3, 0.1, 0.03]:
                for init in [0.3, 0.1, 0.05]:
                    for lr in [0.2, 0.1, 0.05]:
                        # for l2 in [0.00001, 0.0001]:
                        # for l2 in [0.00001, 0.0001, 0.001]:
                        # for t_l2_ in [0.0]:
                        # for t_lr_mul in [1, 0.3, 0.1]:
                        # for t_lr_mul in [1, 0.3]:
                        for t_lr_mul in [0.3]:
                            t_l2 = 0.0
                            init_t = init
                            t_lr = t_lr_mul * lr
                            pretrain_run_no = 0 
                            lstm_norm2 = 100000 
                            net = gen_match_lstm(d_mem=d_mem,init=init,lr=lr,dataset=dataset,l2=l2,lstm_norm2=lstm_norm2,  \
                                                 is_pretrain=True,pretrain_run_no=pretrain_run_no,model_no=model_no,epoch_no=epoch_no)
                            net['log'] = 'log.match.bilstm_tensor_dpool.{0}.d{1}.run{2}.{3}'.format\
                                         (dataset, str(d_mem), str(run), str(idx))
                            net["save_model"] = {"file_prefix": "./model/model."+str(idx),"save_interval": 1000}
                            net["save_activation"] = [{"tag":"Test","file_prefix": \
                                                       "./model/test."+str(idx), \
                                                       "save_interval": 1000, \
                                                       "save_nodes":["x","y","word_rep_seq","l_sentence",\
                                                                     "r_sentence","interaction_rep", \
                                                                     # "interaction_rep_nonlinear",\
                                                                     "dpool_rep","softmax_prob"], \
                                                       "save_iter_num":1}]


                            gen_conf_file(net, '/home/wsx/exp/match/{0}/bilstm_tensor_dpool/run.{1}/'.format(dataset,str(run)) + \
                                               'model.match.bilstm_tensor_dpool.{0}.d{1}.run{2}.{3}'.format\
                                               (dataset, str(d_mem), str(run), str(idx)))
                            idx += 1
