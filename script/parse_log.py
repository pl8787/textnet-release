#-*-coding:utf8-*-
import os

def parse_one_run(lines, tag):
    output = []
    for line in lines:
        if line[1:len(tag)+1] == tag and 'acc' in line:
            output.append(float(line.strip().split()[-1]))

    return output

def split_cv(inFile):
    tag = 'PROCESSING CROSS VALIDATION FOLD'
    begin = end = 0
    result = [] 
    lines = open(inFile).readlines()
    for i, line in enumerate(lines):
        if i == 0:
            continue
        if tag in line:
            end = i
            result.append(lines[begin:end])
            begin = end

    result.append(lines[begin:]) # the last fold

    return result

def parse_one_fold_loss_graph(inFile):
    folds = split_cv(inFile)

    ret = []
    for fold in folds:
        valid = parse_one_run(fold, 'Valid')
        test = parse_one_run(fold, 'Test')
        assert len(valid) == len(test)
        ret.append(test[valid.index(max(valid))])
        break
    fo = open('./tmp.csv', 'w')
    fo.write('Valid,Test\n')
    for i in range(len(valid)):
        fo.write('%f,%f\n' % (valid[i],test[i]))
    return 


def parse_cv_final_acc(inFile):
    print inFile
    if not os.path.exists(inFile):
        print "NOT FOUND"
        return
    folds = split_cv(inFile)
    # print 'num_folds', len(folds)
    assert len(folds) == 10

    valids, tests = [], []
    for fold in folds:
        valid = parse_one_run(fold, 'Valid')
        test = parse_one_run(fold, 'Test')
        assert len(valid) == len(test)
        valids.append(max(valid))
        # print 'max valid:', valids[-1]
        tests.append(test[valid.index(max(valid))])
        # print 'test:', tests[-1]
    print sum(valids)/len(valids), sum(tests)/len(tests)

def parse_max_test(inFile):
    print inFile
    if not os.path.exists(inFile):
        print "NOT FOUND"
        return
    lines = open(inFile).readlines()
    valid = parse_one_run(lines, 'Valid')
    test = parse_one_run(lines, 'Test')
    print max(test)

def parse_tvt(inFile):
    print inFile
    if not os.path.exists(inFile):
        print "NOT FOUND"
        return
    lines = open(inFile).readlines()
    valid = parse_one_run(lines, 'Valid')
    test = parse_one_run(lines, 'Test')
    max_valid = max(valid)
    print max_valid, test[valid.index(max_valid)]

# dir = '/home/wsx/exp/match/bilstm_mlp/run.15/'
# for i in range(12):
#     try:
#         parse_tvt(dir + 'log.match.bilstm_mlp.last.batch10.msrp.d100.'+str(i))
#     except:
#         continue
# exit(0)
# dir = '/home/wsx/exp/gate/lstm/run.7/'
# dir = '/home/wsx/exp/topk_simulation/run.4/'
# for i in range(6):
#     parse_max_test(dir + 'log.cnn.topk.max.w2.d5.'+str(i))
# for i in range(6):
#     parse_max_test(dir + 'log.cnn.topk.ave.w2.d5.'+str(i))
# exit(0)
# for i in range(7):
#     try:
#         parse_max_test(dir + 'log.gate_cnn.topk.max.w2.d50.'+str(i))
#     except:
#         continue
# for i in range(7):
#     try:
#         parse_max_test(dir + 'log.gate_all_cnn.topk.max.w2.d50.'+str(i))
#     except:
#         continue
# exit(0)
# parse_tvt(dir + 'log.gate_mul_cnn.tb_fine.kim.'+str(0))
# parse_tvt(dir + 'log.gate_mul_cnn.tb_binary.kim.'+str(0))
# parse_cv_final_acc(dir + 'log.gate_mul_cnn.mr.kim.'+str(0))
# exit(0)
# dir = '/home/wsx/exp/ccir2015/mr/birnn/run.2/'
dir = '/home/wsx/exp/match/msrp/lstm_sim_dpool/run.1/'
# dir = '/home/wsx/log.tmp/'
# dir = '/home/wsx/exp/match/msrp/cnn/run.1/'
# dir = '/home/wsx/exp/match/msrp/xor_diag_rnn/run.8/'
for i in range(0, 12):
    # try:
        # parse_tvt(dir+'log.match.elem_diag_rnn.msrp.d50.run4.'+str(i))
        # parse_tvt(dir+'log.match.elem_diag_rnn.msrp.d50.run1.'+str(i))
    try:
        parse_tvt(dir+'log.match.lstm_sim_dpool.msrp.d50.run1.'+str(i))
        # parse_cv_final_acc(dir+'log.birnn.mr.d50.run2.'+str(i))
    except:
        print "FAILDED"
        continue
    # parse_cv_final_acc(dir+'log.conv_birnn.max.mr.d50.run1.'+str(i))
    # parse_tvt(dir + 'log.match.xor_diag_rnn.msrp.d50.run8.'+str(i))
    
exit(0)
for i in range(40):
    # parse_cv_final_acc(dir + 'log.conv_bilstm.max.dropbeforepool.mr.d50.'+str(i))
    try:
        # parse_cv_final_acc(dir + 'log.conv_bilstm.max.mr.d50.'+str(i))
        # parse_tvt(dir + 'log.conv_bilstm.max.mr.d50.'+str(i))
        # parse_cv_final_acc(dir + 'log.conv_birnn.max.nodrop.mr.d50.'+str(i))
        parse_cv_final_acc(dir + 'log.conv_bilstm.max.mr.d50.run5.'+str(i))
        # parse_tvt(dir + 'log.match.bilstm_mlp.last.nodrop.batch10.msrp.d50.run1.'+str(i))
    except:
        continue
exit(0)
# for i in range(9):
#     parse_cv_final_acc(dir + 'log.conv_bilstm.max.dropbeforepool.mr.d75.'+str(i))
for i in range(9):
    # parse_tvt(dir + 'log.conv_bilstm.max.dropbeforepool.tb_binary.d50.'+str(i))
    parse_tvt(dir + 'log.conv_bilstm.max.drop2.tb_binary.d50.'+str(i))
# for i in range(9):
#     parse_tvt(dir + 'log.conv_bilstm.max.dropbeforepool.tb_binary.d75.'+str(i))
for i in range(9):
    parse_tvt(dir + 'log.conv_bilstm.max.drop2.tb_fine.d50.'+str(i))
    # parse_tvt(dir + 'log.conv_bilstm.max.dropbeforepool.tb_fine.d50.'+str(i))
# for i in range(9):
#     parse_tvt(dir + 'log.conv_bilstm.max.dropbeforepool.tb_fine.d75.'+str(i))
exit(0)
for i in range(6):
    parse_tvt(dir + 'log.gate_originword_topk_cnn.tb_fine.sum.adagrad.'+str(i))
for i in range(6):
    try:
        parse_tvt(dir + 'log.gate_originword_topk_cnn.tb_binary.sum.adagrad.'+str(i))
    except:
        continue
exit(0)


dir = '/home/wsx/exp/gate/run.17/'
for i in range(6):
    parse_cv_final_acc(dir + 'log.gate_originword_topk_cnn.mr.sum.adagrad.'+str(i))
for i in range(6):
    parse_tvt(dir + 'log.gate_originword_topk_cnn.tb_fine.sum.adagrad.'+str(i))
for i in range(6):
    try:
        parse_tvt(dir + 'log.gate_originword_topk_cnn.tb_binary.sum.adagrad.'+str(i))
    except:
        continue
exit(0)

# dir = '/home/wsx/exp/gate/adagrad/'
# for i in range(6):
#     parse_cv_final_acc(dir + 'log.cnn.mr.max.adagrad.'+str(i))
# for i in range(6):
#     parse_tvt(dir + 'log.cnn.tb_fine.max.adagrad.'+str(i))
# for i in range(6):
#     parse_tvt(dir + 'log.cnn.tb_binary.max.adagrad.'+str(i))
exit(0)
exit(0)
parse_cv_final_acc(dir + 'log.gate_all_mul_cnn.mr.kim.ave.0')
parse_cv_final_acc(dir + 'log.gate_all_mul_cnn.mr.kim.ave.1')
parse_cv_final_acc(dir + 'log.gate_all_mul_cnn.mr.kim.ave.4')
exit(0)

# dir = '/home/wsx/exp/'
# parse_cv_final_acc(dir + 'log.kim')
# parse_cv_final_acc(dir + 'log.mul_cnn.mr.kim')
# exit(0)
# parse_tvt(dir + 'log.tb.kim')
# parse_tvt(dir + 'log.mul_cnn.tb_binary.kim')
# exit(0)
dir = '/home/wsx/exp/gate/run.5/'
parse_cv_final_acc(dir + 'log.gate_mul_cnn.mr.kim.0')
# parse_tvt(dir + 'log.gate_all_mul_cnn.tb_fine.kim.0')
dir = '/home/wsx/exp/gate/run.5/node/'
parse_cv_final_acc(dir + 'log.gate_mul_cnn.mr.kim.1')
parse_cv_final_acc(dir + 'log.gate_mul_cnn.mr.kim.2')
parse_cv_final_acc(dir + 'log.gate_mul_cnn.mr.kim.3')
parse_cv_final_acc(dir + 'log.gate_mul_cnn.mr.kim.4')
parse_cv_final_acc(dir + 'log.gate_mul_cnn.mr.kim.5')

# exit(0)
# parse_tvt(dir + 'log.gate_mul_cnn.tb_binary.kim.0')
# parse_tvt(dir + 'log.gate_mul_cnn.tb_binary.kim.1')
# parse_tvt(dir + 'log.gate_mul_cnn.tb_binary.kim.2')
# parse_tvt(dir + 'log.gate_mul_cnn.tb_binary.kim.3')
# parse_tvt(dir + 'log.gate_mul_cnn.tb_binary.kim.4')
# parse_tvt(dir + 'log.gate_mul_cnn.tb_binary.kim.5')
dir = '/home/wsx/exp/gate/run.5/'
parse_tvt(dir + 'log.gate_all_mul_cnn.tb_binary.kim.0')
dir = '/home/wsx/exp/gate/run.6/'

parse_tvt(dir + 'log.gate_all_mul_cnn.tb_binary.kim.1')
parse_tvt(dir + 'log.gate_all_mul_cnn.tb_binary.kim.2')
parse_tvt(dir + 'log.gate_all_mul_cnn.tb_binary.kim.3')
parse_tvt(dir + 'log.gate_all_mul_cnn.tb_binary.kim.4')
parse_tvt(dir + 'log.gate_all_mul_cnn.tb_binary.kim.5')
# exit(0)
# parse_tvt(dir + 'log.gate_all_mul_cnn.tb_binary.kim.1')
# parse_tvt(dir + 'log.gate_all_mul_cnn.tb_binary.kim.2')
# parse_tvt(dir + 'log.gate_all_mul_cnn.tb_binary.kim.3')
# parse_tvt(dir + 'log.gate_all_mul_cnn.tb_binary.kim.4')
# parse_tvt(dir + 'log.gate_all_mul_cnn.tb_binary.kim.5')

# parse_tvt(dir + 'log.gate_mul_cnn.tb_fine.kim.0')
# parse_tvt(dir + 'log.gate_all_mul_cnn.tb_fine.kim.0')
# parse_tvt(dir + 'log.gate_mul_cnn.tb_binary.kim.0')
# parse_tvt(dir + 'log.gate_mul_cnn.tb_binary.kim.1')
# parse_tvt(dir + 'log.gate_mul_cnn.tb_binary.kim.2')
# parse_tvt(dir + 'log.gate_mul_cnn.tb_binary.kim.3')
# parse_tvt(dir + 'log.gate_mul_cnn.tb_binary.kim.4')
# parse_tvt(dir + 'log.gate_mul_cnn.tb_binary.kim.5')
# parse_tvt(dir + 'log.gate_all_mul_cnn.tb_binary.kim.0')
# parse_tvt(dir + 'log.gate_all_mul_cnn.tb_binary.kim.1')
# parse_tvt(dir + 'log.gate_all_mul_cnn.tb_binary.kim.2')
# parse_tvt(dir + 'log.gate_all_mul_cnn.tb_binary.kim.3')
# parse_tvt(dir + 'log.gate_all_mul_cnn.tb_binary.kim.4')
# parse_tvt(dir + 'log.gate_all_mul_cnn.tb_binary.kim.5')
# parse_cv_final_acc(dir + 'log.gate_mul_cnn.mr.kim.0')
# parse_cv_final_acc(dir + 'log.gate_mul_cnn.mr.kim.1')
# parse_cv_final_acc(dir + 'log.gate_mul_cnn.mr.kim.2')
# parse_cv_final_acc(dir + 'log.gate_mul_cnn.mr.kim.3')
# parse_cv_final_acc(dir + 'log.gate_mul_cnn.mr.kim.4')
# parse_cv_final_acc(dir + 'log.gate_mul_cnn.mr.kim.5')
# print '----'
# parse_cv_final_acc(dir + 'log.gate_all_mul_cnn.mr.kim.0')
# parse_cv_final_acc(dir + 'log.gate_all_mul_cnn.mr.kim.1')
# parse_cv_final_acc(dir + 'log.gate_all_mul_cnn.mr.kim.2')
# parse_cv_final_acc(dir + 'log.gate_all_mul_cnn.mr.kim.3')
# parse_cv_final_acc(dir + 'log.gate_all_mul_cnn.mr.kim.4')
# parse_cv_final_acc(dir + 'log.gate_all_mul_cnn.mr.kim.5')
# parse_cv_final_acc(dir + 'log.gate_cnn.mr.kim.0')
# parse_cv_final_acc(dir + 'log.gate_cnn.mr.kim.1')
# parse_cv_final_acc(dir + 'log.gate_cnn.mr.kim.2')
# parse_cv_final_acc(dir + 'log.gate_cnn.mr.kim.3')
# parse_cv_final_acc(dir + 'log.gate_cnn.mr.kim.4')
# parse_cv_final_acc(dir + 'log.gate_cnn.mr.kim.5')
# parse_cv_final_acc(dir + 'log.gate_cnn.mr.kim.6')
# parse_cv_final_acc(dir + 'log.tb.kim')
# parse_cv_final_acc(dir + 'log.gate.tb.kim')
# parse_cv_final_acc(dir + 'log.kim.nobias')
# parse_cv_final_acc(dir + 'log.tb.kim.nobias.nobatchsize')
# parse_cv_final_acc(dir + 'log.tb.kim.nobias.nobatchsize')
# parse_tvt(dir + 'log.gate_cnn.tb.kim.0')
# parse_tvt(dir + 'log.gate_cnn.tb.kim.2')
# parse_tvt(dir + 'log.gate_cnn.tb.kim.6')
exit(0)
# dir = '/home/wsx/exp/tb/log/run.3/'
for i in range(20):
    # parse_cv_final_acc(dir + 'log.conv_bilstm.max.mr.'+str(i))
    # parse_cv_final_acc(dir + 'log.cnn.max.mr.'+str(i))
    # parse_cv_final_acc(dir + 'log.conv_bilstm.max.mr.'+str(i))
    # parse_cv_final_acc(dir + 'log.gate_cnn.max.mr.'+str(i))
    # parse_cv_final_acc(dir + 'log.cnn.max.tb.'+str(i))
    parse_cv_final_acc(dir + 'log.gate_cnn.max.tb.'+str(i))
    # parse_cv_final_acc(dir + 'log.cnnlstm.max.mr.'+str(i))
    # parse_cv_final_acc(dir + 'log.conv_bilstm.max.mr.'+str(i))
    # parse_cv_final_acc(dir + 'log.cnnlstm.max.mr.'+str(i))
    # parse_tvt(dir + 'log.lstm.last.tb.'+str(i))
    # parse_tvt(dir + 'log.bilstm.max.tb_fine.'+str(i))
    # parse_tvt(dir + 'log.conv_bilstm.max.tb_fine.'+str(i))
    # parse_tvt(dir + 'log.conv_bilstm.max.tb_fine.'+str(i))
    # parse_cv_final(dir + 'log.bilstm.last.tb_fine.'+str(i))
    # parse_cv_final_acc(dir + 'log.cnnlstm.max.mr.'+str(i))
# parse_one_fold_loss_graph(dir + 'log.convlstm.max.mr.d70.0')
