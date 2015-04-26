#-*-coding:utf8-*-
import copy, os

from gen_conf_file import *


def gen_nb(lr = 0.1, l2 = 0., eps = 0.1, log = None):
    d_rep = 50
    d_item_rep = d_rep 
    d_user_rep = d_rep 
    n_hidden = d_rep * 2
    drop_rate = 0.5
    batch_size = 1
    dataset = 'sb'
    if dataset == 'sb':
        train_data_file = '/home/wsx/dl.shengxian/data/pengfei/tafeng_sub.textnet.train.1'
        test_data_file = '/home/wsx/dl.shengxian/data/pengfei/tafeng_sub.textnet.test.1'
        max_session_len = 300
        n_train = 168933
        n_item = 7973
        n_user = 2265
        n_test = n_user 
        n_class = n_item
        max_iter = (10*n_train)/(batch_size);
        test_interval = (n_train/batch_size)/10
        adagrad_reset_iterval = 10000000000 # (n_train/batch_size)/3
        # test_interval = 1

    elif dataset == 'bs':
        assert False
    else:
        assert False

    g_filler = gen_uniform_filter_setting((1./d_rep))
    zero_filler = gen_zero_filter_setting()
    g_updater = gen_adagrad_setting(lr, l2, adagrad_reset_iterval, eps=eps)

    g_layer_setting = {}
    g_layer_setting['no_bias'] = True
    g_layer_setting['phrase_type'] = 2
    g_layer_setting['w_filler'] = g_filler 
    g_layer_setting['b_filler'] = zero_filler
    g_layer_setting['w_updater'] = g_updater
    g_layer_setting['b_updater'] = g_updater

    net = {}
    if log:
        net['log'] = log

    # net['global'] = {}
    # net['global']['wb_updater_ph'] = {'w_updater':copy.deepcopy(g_updater), 'b_updater':copy.deepcopy(g_updater)}
    # net['global']['w_updater_ph'] = {'w_updater':copy.deepcopy(g_updater)}
    net['net_name'] = 'nb_sb'
    net_cfg_train, net_cfg_test = {}, {}
    net['net_config'] = [net_cfg_train, net_cfg_test]
    net_cfg_train["tag"] = "Train"
    net_cfg_train["max_iters"] = max_iter
    net_cfg_train["display_interval"] = 1000
    net_cfg_train["out_nodes"] = ['loss', 'acc']
    net_cfg_test["tag"] = "Valid"
    net_cfg_test["max_iters"] = int(n_test/batch_size) 
    net_cfg_test["display_interval"] = test_interval
    net_cfg_test["out_nodes"] = ['loss', 'acc']

    layers = []
    net['layers'] = layers

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['y', 'ys', 'user', 'items']
    layer['layer_name'] = 'train_data'
    layer['layer_type'] = 73
    layer['tag'] = ["Train"]
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['phrase_type'] = 0
    setting['batch_size'] = batch_size
    setting['data_file'] = train_data_file
    setting['max_session_len'] = max_session_len

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = []
    layer['top_nodes'] = ['y', 'ys', 'user', 'items']
    layer['layer_name'] = 'test_data'
    layer['layer_type'] = 73
    layer['tag'] = ["Valid"]
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['phrase_type'] = 1
    setting['batch_size'] = batch_size 
    setting['data_file'] = test_data_file
    setting['max_session_len'] = max_session_len

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['user']
    layer['top_nodes'] = ['user_rep']
    layer['layer_name'] = 'user_embedding'
    layer['layer_type'] = 21
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['feat_size'] = d_user_rep
    setting['word_count'] = n_user

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['items']
    layer['top_nodes'] = ['item_rep_seq']
    layer['layer_name'] = 'item_embedding'
    layer['layer_type'] = 21
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['feat_size'] = d_item_rep
    setting['word_count'] = n_item

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['item_rep_seq']
    layer['top_nodes'] = ['session_rep']
    layer['layer_name'] = 'wholePooling'
    layer['layer_type'] =  25 
    setting = {"phrase_type":2, "pool_type":"ave"}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['user_rep', 'session_rep']
    layer['top_nodes'] = ['concat_rep']
    layer['layer_name'] = 'concat'
    layer['layer_type'] = 18
    setting = {"phrase_type":2, 'bottom_node_num':2}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['concat_rep']
    layer['top_nodes'] = ['hidden_rep_linear']
    layer['layer_name'] = 'transform'
    layer['layer_type'] = 11 
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['num_hidden'] = n_hidden

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['hidden_rep_linear']
    layer['top_nodes'] = ['hidden_rep_nonlinear']
    layer['layer_name'] = 'activation'
    layer['layer_type'] = 1 # relu 1 sigmoide 2 tanh 3
    setting = {"phrase_type":2}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['hidden_rep_nonlinear']
    layer['top_nodes'] = ['drop_rep']
    layer['layer_name'] = 'dropout'
    layer['layer_type'] =  13
    setting = {'phrase_type':2, 'rate':drop_rate}
    layer['setting'] = setting

    layer = {}
    layers.append(layer) 
    layer['bottom_nodes'] = ['drop_rep']
    layer['top_nodes'] = ['softmax_ret']
    layer['layer_name'] = 'softmax_fullconnect'
    layer['layer_type'] = 11 
    setting = copy.deepcopy(g_layer_setting)
    layer['setting'] = setting
    setting['num_hidden'] = n_class
    # setting['w_filler'] = zero_filler
    print 'ORC: softmax init non_zero.'

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
    layer['bottom_nodes'] = ['softmax_ret', 'ys']
    layer['top_nodes'] = ['acc']
    layer['layer_name'] = 'accuracy'
    layer['layer_type'] = 56 
    setting = {'phrase_type':2, 'topk':5}
    layer['setting'] = setting

    return net

local_dir = '/home/wsx/exp/nb/log/run.2/'
conf_idx = 0
for lr in [0.1, 0.06, 0.03]:
    # for l2 in [0., 0.000001, 0.00001]:
    for eps in [1, 0.1, 0.01]:
       conf_idx += 1
       log = 'log.' + str(conf_idx)
       net = gen_nb(lr=lr, l2=0, eps=eps, log=log)
       gen_conf_file(net, local_dir + 'cfg.' + str(conf_idx))
