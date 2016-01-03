#-*-coding:utf8-*-
import copy, os

from gen_conf_file import *
from dataset_cfg import *

t_l2 = 0.
t_lr = 0.
init_t = 0.0
f_gate_bias_init = 0.0

def gen_match_lstm(d_mem, init, lr, dataset, l2):
    is_use_mlp = False 
    print "ORC: using MLP"

    # print "ORC: left & right lstm share parameters"
    net = {}

    ds = DatasetCfg(dataset)
    g_filler    = gen_uniform_filler_setting(init)
    zero_filler = gen_zero_filler_setting()
    embed_filler = {'init_type':6, 'vec_len':50}
    t_updater   = gen_adagrad_setting(lr = t_lr, l2 = t_l2, batch_size = ds.train_batch_size)
    g_updater   = gen_adagrad_setting(lr = lr, l2 = l2, batch_size = ds.train_batch_size)
    print "ORC: using small learning rate for embedding"
    zero_l2_updater   = gen_adagrad_setting(lr = lr*0.1, batch_size = ds.train_batch_size)
    # g_updater   = gen_sgd_setting(lr = lr, l2 = l2, batch_size = ds.train_batch_size)
    # zero_l2_updater   = gen_sgd_setting(lr = lr, batch_size = ds.train_batch_size)

    g_layer_setting = {}
    g_layer_setting['no_bias'] = False

    g_layer_setting['t_filler'] = g_filler 
    g_layer_setting['w_filler'] = g_filler 
    g_layer_setting['b_filler'] = zero_filler
    g_layer_setting['t_updater'] = t_updater
    g_layer_setting['w_updater'] = g_updater
    g_layer_setting['b_updater'] = g_updater

    net['net_name'] = 'match_lstmd2_tensor_dpool'
    net['need_reshape'] = False
    net_cfg_train, net_cfg_valid, net_cfg_test = {}, {}, {}
    net['net_config'] = [net_cfg_train, net_cfg_valid, net_cfg_test]
    net_cfg_train["tag"] = "Train"
    net_cfg_train["max_iters"] = ds.train_max_iters * 2
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
    print "ORC: WITH embedding used"
    setting['embedding_file'] = ds.embedding_file
    print "ORC: update all words"
    # setting['update_indication_file'] = ds.update_indication_file
    setting['feat_size'] = ds.d_word_rep
    setting['word_count'] = ds.vocab_size
    print "ORC: not use l2 for embedding"
    setting['w_filler'] = embed_filler
    setting['w_updater'] = zero_l2_updater

    layer = {}
    layers.append(layer)
    layer['bottom_nodes'] = ['word_rep_seq']
    layer['top_nodes'] = ['l_sentence', 'r_sentence']
    layer['layer_name'] = 'sentence_split'
    layer['layer_type'] = 20 
    setting = {}
    layer['setting'] = setting

    # layer = {}
    # layers.append(layer) 
    # layer['bottom_nodes'] = ['l_sentence', 'r_sentence']
    # layer['top_nodes'] = ['dot_similarity']
    # layer['layer_name'] = 'match'
    # layer['layer_type'] = 23 
    # print "ORC: use MUL operation for similarity"
    # layer['setting'] = {'op':'mul'}

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['l_sentence', 'r_sentence']
    layer['top_nodes'] = ['interaction_rep']
    layer['layer_name'] = 'match_tensor'
    layer['layer_type'] = 1001
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    print 'ORC: tensor dim:', d_mem
    setting['d_hidden'] = d_mem 
    setting['t_l2'] = t_l2
    setting['is_use_linear'] = True
    setting['t_updater'] = t_updater
    setting['w_updater'] = t_updater
    setting['t_filler'] = gen_uniform_filler_setting(init_t)
    setting['w_filler'] = gen_uniform_filler_setting(init_t)

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
    layer['bottom_nodes'] = ['interaction_rep_nonlinear']
    # layer['bottom_nodes'] = ['dot_similarity']
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
    setting['f_gate_bias_init'] = f_gate_bias_init
    # print "ORC: LSTM output gate:", 1.
    setting['o_gate_bias_init'] = 0.

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['lstm_d2_input']
    layer['top_nodes'] = ['match_matrix_br2lt'] # left top to bottom right
    layer['layer_name'] = 'lstm_d2_br2lt'
    layer['layer_type'] = 10005
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    print "ORC: LSTM dim:", d_mem
    setting['d_mem'] = d_mem
    setting['reverse'] = True
    setting['f_gate_bias_init'] = f_gate_bias_init
    # print "ORC: LSTM output gate:", 1.
    setting['o_gate_bias_init'] = 0.

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
    
    # 要把轴再换过来，因为dynamic poolin的时候，轴对不上
    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['match_matrix']
    # ayer['bottom_nodes'] = ['match_matrix_lt2br']
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

    # 这个是之前为了对比cnn的吧
    # layer = {}
    # layers.append(layer)
    # layer['bottom_nodes'] = ['interaction_rep_nonlinear']
    # layer['top_nodes'] = ['match_matrix_conv']
    # layer['layer_name'] = 'convolution'
    # layer['layer_type'] = 14
    # setting = copy.deepcopy(g_layer_setting)
    # layer['setting'] = setting
    # setting['channel_out'] = d_mem
    # setting['kernel_x'] = 4 
    # setting['kernel_y'] = 4
    # setting['pad_x'] = 2
    # setting['pad_y'] = 2
    # setting['no_bias'] = True
    # setting['stride'] = 1
    # setting['d1_var_len'] = False

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['match_matrix_output', 'l_sentence', 'r_sentence']
    layer['top_nodes'] = ['dpool_rep']
    layer['layer_name'] = 'dynamic_pooling'
    layer['layer_type'] = 43
    layer['setting'] = {'row':5, 'col':5, 'interval':1}

    # layer = {}
    # layers.append(layer) 
    # layer['bottom_nodes'] = ['match_matrix_output', 'l_sentence', 'r_sentence']
    # layer['top_nodes'] = ['dpool_rep']
    # layer['layer_name'] = 'top_k_pooling'
    # layer['layer_type'] = 10002
    # layer['setting'] = {'k':5}

    if is_use_mlp:
        layer = {}
        layers.append(layer) 
        layer['bottom_nodes'] = ['dpool_rep']
        layer['top_nodes'] = ['hidden_trans']
        layer['layer_name'] = 'mlp_hidden'
        layer['layer_type'] = 11 
        setting = copy.deepcopy(g_layer_setting)
        layer['setting'] = setting
        setting['num_hidden'] = 128

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
    setting['num_hidden'] = 2
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

    return net

run = 8 
l2 = 0.
for dataset in ['paper']:
    for d_mem in [20]:
        idx = 0
        for embedding_lr_mul in [1.0]:
            for gate_bias_init in [-0.4]:
                f_gate_bias_init = gate_bias_init
                for init in [0.15, 0.05]:
                    for lr in [0.3, 0.1, 0.05]:
                        # for t_l2_ in [0.0]:
                        # for t_lr_mul in [1, 0.3, 0.1]:
                        for t_init_mul in [1]:
                            t_lr_mul = 1
                            t_l2 = 0.0
                            init_t = init * t_init_mul
                            t_lr = t_lr_mul * lr
                            net = gen_match_lstm(d_mem=d_mem,init=init,lr=lr,dataset=dataset,l2=l2)
                            net['log'] = 'log.match.lstmd2.{0}.d{1}.run{2}.{3}'.format\
                                         (dataset, str(d_mem), str(run), str(idx))
                            # net["save_model"] = {"file_prefix": "./model/model."+str(idx),"save_interval": 1000}
                            # net["save_activation"] = [{"tag":"Test","file_prefix": \
                            #                            "./model/test."+str(idx), \
                            #                            "save_interval": 1000, \
                            #                            "save_nodes":["x","y","word_rep_seq","l_sentence",\
                            #                                          "r_sentence","interaction_rep", \
                            #                                          "interaction_rep_nonlinear",\
                            #                                          "match_matrix",\
                            #                                          "dpool_rep","softmax_prob"], \
                            #                            "save_iter_num":1}]


                            gen_conf_file(net, '/home/wsx/exp/match/{0}/lstmd2/run.{1}/'.format(dataset,str(run)) + \
                                               'model.match.lstmd2.{0}.d{1}.run{2}.{3}'.format\
                                               (dataset, str(d_mem), str(run), str(idx)))
                            idx += 1
