
class DatasetCfg:
    def __init__(self, dataset):
        if dataset == 'mr':
            self.train_data_file = '/home/wsx/data/movie_review/lstm.train.nopad'
            self.valid_data_file = '/home/wsx/data/movie_review/lstm.valid.nopad'
            self.test_data_file  = '/home/wsx/data/movie_review/lstm.test.nopad'
            self.embedding_file  = '/home/wsx/data/movie_review/word_rep_w2v'

            self.dp_rate = 0.5
            self.batch_size = 50
            self.max_doc_len = 100
            self.vocab_size = 18766
            self.num_class = 2
            self.d_word_rep = 300 

            self.n_train = 1067 * 8
            self.n_valid = 1067
            self.n_test = 1067
        elif dataset == 'tb_fine':
            self.train_data_file = '/home/wsx/data/treebank/train.seq.allnode.unique.fine.shuffle'
            self.valid_data_file = '/home/wsx/data/treebank/dev.seq.fine'
            self.test_data_file  = '/home/wsx/data/treebank/test.seq.fine'
            self.embedding_file  = '/home/wsx/data/treebank/treebank.embed.glove'

            self.dp_rate = 0.5
            self.batch_size = 50
            self.max_doc_len = 100
            self.vocab_size = 21701
            self.num_class = 5
            self.d_word_rep = 300 

            self.n_train = 159247
            self.n_valid = 1101
            self.n_test = 2210 
        elif dataset == 'tb_binary':
            self.train_data_file = '/home/wsx/data/treebank/train.seq.allnode.unique.binary.shuffle'
            self.valid_data_file = '/home/wsx/data/treebank/dev.seq.binary'
            self.test_data_file  = '/home/wsx/data/treebank/test.seq.binary'
            self.embedding_file  = '/home/wsx/data/treebank/treebank.embed.glove'

            self.dp_rate = 0.5
            self.batch_size = 50
            self.max_doc_len = 100
            self.vocab_size = 21701
            self.num_class = 2
            self.d_word_rep = 300 

            self.n_train = 67349
            self.n_valid = 872 
            self.n_test = 1821
        elif dataset == 'simulation':
            self.train_data_file = '/home/wsx/dl.shengxian/data/simulation/neg.gen.train'
            self.test_data_file  = '/home/wsx/dl.shengxian/data/simulation/neg.gen.train'
            self.n_test = 300 
            self.embedding_file = ''
            self.max_doc_len = 100
            self.vocab_size = 2000

            self.num_class = 2
            self.d_word_rep = 30
            self.batch_size = 1
        else:
            assert False
