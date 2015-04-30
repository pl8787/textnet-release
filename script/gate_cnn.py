#-*-coding:utf8-*-
import copy, os

from gen_conf_file import *
from dataset_cfg import *



def gen_gate_cnn(d_mem, init, window, lr, l2 = 0., l2_gate = 0.):
    net = {}
    # dataset = 'tb_binary'
    # norm2 = 9
    # dataset = 'mr'
    dataset = 'simulation_topk'
    if dataset == 'mr':
        net['cross_validation'] = 10

    ds = DatasetCfg(dataset)
    g_filler = gen_uniform_filter_setting(init)
    conv_w_filler = gen_uniform_filter_setting(0.01)
    zero_filler = gen_zero_filter_setting()
    # g_updater = gen_adadelta_setting(batch_size = ds.batch_size)
    g_updater = gen_adagrad_setting(lr = lr, l2 = l2, batch_size = ds.batch_size)
    # norm2_updater = gen_adadelta_setting(batch_size = ds.batch_size, norm2 = norm2)

    g_layer_setting = {}
    g_layer_setting['no_bias'] = True
    g_layer_setting['phrase_type'] = 2
    g_layer_setting['w_filler'] = g_filler 
    g_layer_setting['u_filler'] = g_filler
    g_layer_setting['b_filler'] = zero_filler
    g_layer_setting['w_updater'] = g_updater
    g_layer_setting['u_updater'] = g_updater
    g_layer_setting['b_updater'] = g_updater

    net['net_name'] = 'cnn'
    net_cfg_train, net_cfg_valid, net_cfg_test = {}, {}, {}
    net['net_config'] = [net_cfg_train, net_cfg_valid, net_cfg_test]
    net_cfg_train["tag"] = "Train"
    net_cfg_train["max_iters"] = (ds.n_train * 100)/ ds.batch_size 
    net_cfg_train["display_interval"] = (ds.n_train/ds.batch_size)/300
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
    layer['top_nodes'] = ['conv_seq']
    layer['layer_name'] = 'conv'
    layer['layer_type'] = 14
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['w_filler'] = conv_w_filler
    setting['d1_var_len'] = True
    setting['channel_out'] = d_mem
    setting['pad_y'] = window-1
    setting['kernel_y'] = window
    setting['kernel_x'] = ds.d_word_rep

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['conv_seq']
    layer['top_nodes'] = ['conv_activation']
    layer['layer_name'] = 'nonlinear'
    layer['layer_type'] = 1 
    setting = {"phrase_type":2}
    layer['setting'] = setting
        
    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['conv_activation']
    layer['top_nodes'] = ['conv_ret_trans']
    layer['layer_name'] = 'conv_ret_transform'
    layer['layer_type'] =  32 
    setting = {"phrase_type":2}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['conv_ret_trans']
    layer['top_nodes'] = ['gate_result']
    layer['layer_name'] = 'gate'
    print "ORC: gate on one"
    layer['layer_type'] =  33
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['no_bias'] = True
    setting['w_filler'] = zero_filler
    setting['w_updater']['l2'] = l2_gate
    
    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['gate_result']
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
    print "ORC, dropout rate 0.5"
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
    setting['num_hidden'] = ds.num_class
    setting['w_filler'] = zero_filler
    print "ORC, without using norm 2"
    # setting['w_updater'] = norm2_updater

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

    # gen_conf_file(net, '../bin/conv_lstm_simulation.model')

    return net
idx = 0
for d_mem in [50]:
    for window in [2]:
        for lr in [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]:
            net = gen_gate_cnn(d_mem = d_mem, init = 0.01, window = window, lr=lr)
            net['log'] = 'log.gate_cnn.topk.max.w2.d50.' + str(idx)
            gen_conf_file(net, '/home/wsx/exp/topk_simulation/run.2/model.gate_cnn.topk.max.w2.d50.'+str(idx))#  + str(idx))
            idx += 1
 
exit(0)

idx = 0
init = 0.01
for d_mem in [100]:
    for l2_gate in [0., 0.000001, 0.000003, 0.00001, 0.00003, 0.0001, 0.0003]:
        net = gen_gate_cnn(d_mem = d_mem, init = init, window = 5, l2_gate = l2_gate)
        net['log'] = 'log.gate_cnn.mr.kim.' + str(idx)
        gen_conf_file(net, '/home/wsx/exp/gate/run.4/gate_cnn.mr.model.kim.' + str(idx))
        idx += 1
        # # for _, init in enumerate([0.3, 0.1, 0.03, 0.01, 0.003]):
        # for _, init in enumerate([0.3]):
        # # for _, init in enumerate([0.3]):
        #     net = gen_gate_cnn(d_mem = d_mem, init = init, window = window)
        #     # net['log'] = 'log.gate_cnn.max.mr.' + str(idx)
        #     gen_conf_file(net, '/home/wsx/exp/tb/gate_cnn.tb.model.kim' + str(idx))
        #     idx += 1
        #     # gen_conf_file(net, 'gate_cnn.model.tb.' + str(i))
        # os.system("../bin/textnet ../bin/cnn_lstm_mr.model")
        # os.system("../bin/textnet ../bin/conv_lstm_simulation.model > ../bin/simulation/neg.gen.train.{0}".format(d_mem))
