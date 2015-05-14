#-*-coding:utf8-*-
import copy, os

from gen_conf_file import *
from dataset_cfg import *

def gen_conv_bilstm(d_mem, init, l2, lr, dataset, batch_size):
    net = {}
    # dataset = 'tb_fine'
    if dataset == 'mr':
        net['cross_validation'] = 10

    ds = DatasetCfg(dataset)
    # ds.batch_size = batch_size
    ds.train_batch_size = 40
    # ds.train_batch_size = batch_size
    print "ds.train_batch_size:", ds.train_batch_size
    g_filler = gen_uniform_filter_setting(init)
    zero_filler = gen_zero_filter_setting()
    # print "Batch size, not use batch_size in ADA_DELTA"
    # g_updater = gen_adadelta_setting(l2 = l2)
    # zero_l2_updater = gen_adadelta_setting(l2 = 0.)
    g_updater = gen_adadelta_setting(l2 = l2, batch_size = batch_size)
    zero_l2_updater = gen_adadelta_setting(l2 = 0., batch_size = batch_size)
    # g_updater = gen_adagrad_setting(lr = lr, l2 = 0., batch_size = ds.train_batch_size)

    g_layer_setting = {}
    g_layer_setting['no_bias'] = True
    g_layer_setting['w_filler'] = g_filler 
    g_layer_setting['u_filler'] = g_filler
    g_layer_setting['b_filler'] = zero_filler
    g_layer_setting['w_updater'] = g_updater
    g_layer_setting['u_updater'] = g_updater
    g_layer_setting['b_updater'] = g_updater

    net['net_name'] = 'conv_bilstm'
    net['need_reshape'] = True
    net_cfg_train, net_cfg_valid, net_cfg_test = {}, {}, {}
    net['net_config'] = [net_cfg_train, net_cfg_valid, net_cfg_test]
    net_cfg_train["tag"] = "Train"
    net_cfg_train["max_iters"] = (ds.n_train * 10)/ ds.train_batch_size 
    net_cfg_train["display_interval"] = (ds.n_train/ds.train_batch_size)/50
    net_cfg_train["out_nodes"] = ['acc']
    net_cfg_valid["tag"] = "Valid"
    net_cfg_valid["max_iters"] = int(ds.n_valid/ds.valid_batch_size) 
    net_cfg_valid["display_interval"] = (ds.n_train/ds.train_batch_size)/3
    net_cfg_valid["out_nodes"] = ['acc']
    net_cfg_test["tag"] = "Test"
    net_cfg_test["max_iters"] = int(ds.n_test/ds.test_batch_size) 
    net_cfg_test["display_interval"] = (ds.n_train/ds.train_batch_size)/3
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
    setting['batch_size'] = ds.train_batch_size
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
    setting['batch_size'] = ds.valid_batch_size 
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
    setting['batch_size'] = ds.test_batch_size 
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
    print "ORC: without weight decay on word embedding"
    setting['w_updater'] = zero_l2_updater
    setting['embedding_file'] = ds.embedding_file
    setting['feat_size'] = ds.d_word_rep
    setting['word_count'] = ds.vocab_size

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
    # ds.dp_rate = 0.
    print "ORC, dp_rate:", ds.dp_rate
    setting = {'rate':ds.dp_rate}
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

run = 15 
for dataset in ['mr']:
    for d_mem in [50]:
        idx = 0
        for init in [0.1]:
            for l2 in [0.00001, 0.0001]:# , 0.00001, 0.0001, 0.001]:
                for batch_size in [2, 5, 10, 20, 40]:
                    lr = 0.
                    net = gen_conv_bilstm(d_mem = d_mem, init = init, lr =lr, dataset=dataset, l2 = l2, batch_size=batch_size)
                    net['log'] = 'log.conv_bilstm.max.{0}.d{1}.run{2}.{3}'.format(dataset, str(d_mem), str(run),str(idx))
                    # gen_conf_file(net, '/home/wsx/exp/tb/log/run.3/bilstm.max.tb_fine.model.' + str(idx))
                    gen_conf_file(net, '/home/wsx/exp/ccir2015/run.15/model.conv_bilstm.max.{0}.d{1}.run{2}.{3}'.format(dataset, str(d_mem), str(run), str(idx)))
                    idx += 1
 
# idx = 0
# for d_mem in [25, 50, 70]:
#     # for init in [0.3, 0.1, 0.03, 0.01]:
#     for init in [0.3, 0.1, 0.03]:
#         for l2 in [0.000001, 0.000003, 0.00001, 0.00003]:
#             net = gen_conv_bilstm(d_mem = d_mem, init = init, l2=l2)
#             net['log'] = 'log.conv_bilstm.max.mr.' + str(idx)
#             # gen_conf_file(net, '/home/wsx/exp/tb/log/run.4/conv_bilstm.max.tb_fine.model.' + str(idx))
#             gen_conf_file(net, '/home/wsx/exp/mr/log/run.10/conv_bilstm.max.mr.model.' + str(idx))
#             idx += 1
#     # os.system("../bin/textnet ../bin/cnn_lstm_mr.model")
#     # os.system("../bin/textnet ../bin/conv_lstm_simulation.model > ../bin/simulation/neg.gen.train.{0}".format(d_mem))
