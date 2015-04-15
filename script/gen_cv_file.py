#-*-coding:utf8-*-
import random


def gen_cv_tvts(all, n_fold):
    def get_one(all, n_fold, fold_idx):
        fold_size = int(len(all) / n_fold)
        test_fold = fold_idx
        if test_fold == n_fold-1:
            test = all[test_fold*fold_size :]
        else: 
            test = all[test_fold*fold_size : (test_fold+1)*fold_size]
        valid_fold = (fold_idx+1) % n_fold
        if valid_fold == n_fold-1:
            valid = all[valid_fold*fold_size :]
        else: 
            valid = all[valid_fold*fold_size : (valid_fold+1)*fold_size]
        train = []
        for i in range(n_fold):
            if i == test_fold or i == valid_fold:
                continue
            if i == n_fold-1:
                train += all[i*fold_size :]
            else:
                train += all[i*fold_size : (i+1)*fold_size]
        assert len(train) + len(valid) + len(test) == len(all)
        return (train, valid, test)

    random.shuffle(all)
    tvts = []
    for i in range(n_fold):
      tvts.append(get_one(all, n_fold, i))
    return tvts

def output(data, file):
    fo = open(file, 'w')
    for line in data:
        fo.write(line)
    fo.close()

# one line is a example
def main(allFile, n_fold, trainFile, validFile, testFile):
    all = open(allFile).readlines()
    tvts = gen_cv_tvts(all, n_fold)
    for i,tvt in enumerate(tvts):
        output(tvt[0], trainFile+'.'+str(i))
        output(tvt[1], validFile+'.'+str(i))
        output(tvt[2], testFile +'.'+str(i))
    return

if __name__ == '__main__':
    dir = '/home/wsx/dl.shengxian/data/mr/'
    all = dir + 'lstm.all.nopad'
    train = dir + 'lstm.train.nopad'
    valid = dir + 'lstm.valid.nopad'
    test = dir + 'lstm.test.nopad'
    main(all, 10, train, valid, test)
