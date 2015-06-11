
class DatasetCfg:
    def __init__(self, dataset):
        if dataset == 'mr':
            self.train_data_file = '/home/wsx/data/movie_review/lstm.train.nopad'
            self.valid_data_file = '/home/wsx/data/movie_review/lstm.valid.nopad'
            self.test_data_file  = '/home/wsx/data/movie_review/lstm.test.nopad'
            self.embedding_file  = '/home/wsx/data/movie_review/word_rep_w2v'

            self.dp_rate = 0.5
            self.batch_size = 50
            self.train_batch_size = 50
            self.valid_batch_size = 10
            self.test_batch_size = 10
            self.max_doc_len = 56
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
            # self.batch_size = 200 
            self.train_batch_size = 20 
            self.valid_batch_size = 10 
            self.test_batch_size  = 10 
            self.max_doc_len = 56
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
            # self.batch_size = 200
            self.train_batch_size = 20 
            self.valid_batch_size = 10 
            self.test_batch_size  = 10 
            self.max_doc_len = 56
            self.vocab_size = 21701
            self.num_class = 2
            self.d_word_rep = 300 

            self.n_train = 67349
            self.n_valid = 872 
            self.n_test = 1821
        elif dataset == 'trec':
            self.train_data_file = '/home/wsx/data/trec/train'
            self.valid_data_file = '/home/wsx/data/trec/valid'
            self.test_data_file  = '/home/wsx/data/trec/test'
            self.embedding_file  = '/home/wsx/data/trec/word.rep'

            self.dp_rate = 0.5
            self.batch_size = 50
            self.train_batch_size = 50
            self.valid_batch_size = 50
            self.test_batch_size = 50
            self.max_doc_len = 40 
            self.vocab_size = 9593 
            self.num_class = 6
            self.d_word_rep = 300 

            self.n_train = 4952
            self.n_valid = 500
            self.n_test = 500
        elif dataset == 'msrp_char':
            self.train_data_file = '/home/wsx/data/msrp/train.char'
            self.valid_data_file = '/home/wsx/data/msrp/valid.char'
            self.test_data_file  = '/home/wsx/data/msrp/test.char'
            
            self.max_doc_len = 225
            self.min_doc_len = 1 
            self.vocab_size = 128

            self.dp_rate = 0.5
            self.num_class = 2
            self.d_word_rep = 100
            self.batch_size = 50
            self.train_batch_size = 50
            self.valid_batch_size = 50
            self.test_batch_size  = 50 
            self.n_train = 7152
            self.n_valid = 500
            self.n_test = 1725
            self.train_display_interval = 1 
            self.valid_display_interval = 100
            self.test_display_interval  = 100 
            self.train_max_iters = 5000
            self.valid_max_iters = self.n_valid/ self.valid_batch_size
            self.test_max_iters  = self.n_test / self.test_batch_size
        elif dataset == 'msrp':
            self.train_data_file = '/home/wsx/data/msrp/msr_paraphrase_num_local_train_wid_dup.txt'
            self.valid_data_file = '/home/wsx/data/msrp/msr_paraphrase_num_local_valid_wid.txt'
            self.test_data_file  = '/home/wsx/data/msrp/msr_paraphrase_num_test_wid.txt'
            self.embedding_file  = '/home/wsx/data/msrp/msrp.embed'
            self.update_indication_file = '/home/wsx/data/msrp/wikicorp_num_50_msr_ind.txt'
            
            self.max_doc_len = 33
            self.min_doc_len = 5 
            self.vocab_size = 15586

            self.dp_rate = 0.5
            self.num_class = 2
            self.d_word_rep = 50
            self.batch_size = 50
            self.train_batch_size = 50
            self.valid_batch_size = 50
            self.test_batch_size  = 50 
            self.n_train = 7152
            self.n_valid = 500
            self.n_test = 1725
            self.train_display_interval = 1 
            self.valid_display_interval = 100
            self.test_display_interval  = 100 
            self.train_max_iters = 5000
            self.valid_max_iters = self.n_valid/ self.valid_batch_size
            self.test_max_iters  = self.n_test / self.test_batch_size
        elif dataset == 'wiki':
            self.data_dir = '/home/wsx/data/wiki/'
            self.train_data_file = self.data_dir + 'wiki.train.with_msrp'
            self.valid_data_file = self.data_dir + 'wiki.valid'
            self.test_data_file  = self.data_dir + 'msrp.sentence.valid'
            self.embedding_file  = self.data_dir + 'wiki.embed'
            self.update_indication_file = self.data_dir + 'wiki.ind'
            self.word_class_file = self.data_dir + 'id2class'
            self.word_freq_file = self.data_dir + 'word_freq'
            
            self.max_doc_len = 50
            self.min_doc_len = 5
            self.vocab_size = 177859 # without orc_unknown

            self.dp_rate = 0.
            self.d_word_rep = 50
            self.batch_size = 32
            self.train_batch_size = 32
            self.valid_batch_size = 32
            self.test_batch_size  = 32
            self.n_train = 924735
            self.n_valid = 94802
            self.n_test = 1000
            self.train_display_interval = 10 
            self.valid_display_interval = 1000 
            self.test_display_interval  = 1000 
            self.train_max_iters = (self.n_train/self.train_batch_size) * 10
            self.valid_max_iters = (self.n_valid/5)/self.valid_batch_size
            self.test_max_iters  = self.n_test /self.test_batch_size
        elif dataset == 'paper':
            self.train_data_file = '/home/wsx/data/PaperData/relation.train.wid.txt'
            self.valid_data_file = '/home/wsx/data/PaperData/relation.valid.wid.txt'
            self.test_data_file  = '/home/wsx/data/PaperData/relation.test.wid.txt'
            self.embedding_file  = '/home/wsx/data/PaperData/wikicorp_50_english_norm.txt'
            self.update_indication_file = ''
            
            self.max_doc_len = 32 
            self.min_doc_len = 4
            self.vocab_size = 256017

            self.dp_rate = 0.5
            self.num_class = 2
            self.d_word_rep = 50
            self.batch_size = 128 
            self.train_batch_size = 128 
            self.valid_batch_size = 128
            self.test_batch_size  = 128 

            # self.n_train = 6152
            self.n_valid = 119829
            self.n_test = 119883
            self.train_display_interval = 1 
            self.valid_display_interval = 2000
            self.test_display_interval  = 2000 

            self.train_max_iters = 40000
            self.valid_max_iters = self.n_valid/ self.valid_batch_size
            self.test_max_iters  = self.n_test / self.test_batch_size
        elif dataset == 'relation':
            self.train_data_file = '/home/wsx/data/relation/relation.train.wid.txt'
            self.valid_data_file = '/home/wsx/data/relation/relation.valid.wid.txt'
            self.test_data_file  = '/home/wsx/data/relation/relation.test.wid.txt'
            self.embedding_file  = '/home/wsx/data/relation/wikicorp_50_english_norm.txt'
            self.update_indication_file = '/home/wsx/data/relation/wikicorp_50_english_ind.txt'
            
            self.max_doc_len = 32 
            self.min_doc_len = 4
            self.vocab_size = 415472

            self.dp_rate = 0.5
            self.num_class = 2
            self.d_word_rep = 50
            self.batch_size = 32 
            self.train_batch_size = 32
            self.valid_batch_size = 32
            self.test_batch_size  = 32 
            self.train_display_interval = 1 
            self.valid_display_interval = 2000
            self.test_display_interval  = 2000 
            self.train_max_iters = 200000
            self.valid_max_iters = 1000
            self.test_max_iters  = 1000 
        elif dataset == 'relation_dep':
            self.train_data_file = '/home/wsx/data/relation_dep/relation.train.wid.txt'
            self.valid_data_file = '/home/wsx/data/relation_dep/relation.valid.wid.txt'
            self.test_data_file  = '/home/wsx/data/relation_dep/relation.test.wid.txt'
            self.embedding_file  = '/home/wsx/data/relation_dep/wikicorp_50_english_norm.txt'
            self.update_indication_file = '/home/wsx/data/relation_dep/wikicorp_50_english_ind.txt'
            
            self.max_doc_len = 32 
            self.min_doc_len = 4
            self.vocab_size = 415472

            self.dp_rate = 0.5
            self.num_class = 2
            self.d_word_rep = 50
            self.batch_size = 32 
            self.train_batch_size = 32
            self.valid_batch_size = 32
            self.test_batch_size  = 32 
            self.train_display_interval = 1 
            self.valid_display_interval = 2000
            self.test_display_interval  = 2000 
            self.train_max_iters = 200000
            self.valid_max_iters = 1000
            self.test_max_iters  = 1000 
        elif dataset == 'relation_dep_100':
            self.train_data_file = '/home/wsx/data/relation_dep_100/relation.train.wid.txt'
            self.valid_data_file = '/home/wsx/data/relation_dep_100/relation.valid.wid.txt'
            self.test_data_file  = '/home/wsx/data/relation_dep_100/relation.test.wid.txt'
            self.embedding_file  = '/home/wsx/data/relation_dep_100/wikicorp_100_english_norm.txt'
            self.update_indication_file = '/home/wsx/data/relation_dep_100/wikicorp_100_english_ind.txt'
            
            self.max_doc_len = 32 
            self.min_doc_len = 4
            self.vocab_size = 415472

            self.dp_rate = 0.5
            self.num_class = 2
            self.d_word_rep = 100
            self.batch_size = 32 
            self.train_batch_size = 32
            self.valid_batch_size = 32
            self.test_batch_size  = 32 
            self.train_display_interval = 1 
            self.valid_display_interval = 2000
            self.test_display_interval  = 2000 
            self.train_max_iters = 200000
            self.valid_max_iters = 1000
            self.test_max_iters  = 1000 
        elif dataset == 'simulation':
            self.train_data_file = '/home/wsx/dl.shengxian/data/simulation/neg.gen.train'
            self.valid_data_file  = '/home/wsx/dl.shengxian/data/simulation/neg.gen.train'
            self.test_data_file  = '/home/wsx/dl.shengxian/data/simulation/neg.gen.test'
            self.embedding_file = ''
            
            self.max_doc_len = 20
            self.vocab_size = 2000

            self.dp_rate = 0.5
            self.num_class = 2
            self.d_word_rep = 20
            self.batch_size = 1
            self.train_batch_size = 1
            self.valid_batch_size = 1
            self.test_batch_size = 1
            self.n_train = 300
            self.n_valid = 300
            self.n_test = 200
        elif dataset == 'simulation_topk':
            self.train_data_file = '/home/wsx/dl.shengxian/data/simulation/gen.train.topk'
            self.valid_data_file  = '/home/wsx/dl.shengxian/data/simulation/gen.train.topk'
            self.test_data_file  = '/home/wsx/dl.shengxian/data/simulation/gen.test.topk'
            self.embedding_file = ''
            
            self.max_doc_len = 10
            self.vocab_size = 10000

            self.dp_rate = 0.5
            self.num_class = 2
            self.d_word_rep = 30
            self.batch_size = 10
            self.train_batch_size = 1
            self.valid_batch_size = 1
            self.test_batch_size = 1
            self.n_train = 3000
            self.n_valid = 3000
            self.n_test = 2000
        elif dataset == 'test_lm':
            self.train_data_file = '/home/wsx/data/test/test_lm/train.txt'
            self.valid_data_file = '/home/wsx/data/test/test_lm/train.txt'
            self.test_data_file = '/home/wsx/data/test/test_lm/train.txt'
            self.embedding_file = ''
            self.update_indication_file = ''
            self.max_doc_len = 5
            self.vocab_size = 8  
            # self.negative_num = 1
            # self.position_num = 1

            self.d_word_rep = 5
            self.batch_size = 1
            self.train_batch_size = 1
            self.valid_batch_size = 1
            self.test_batch_size = 1
            self.n_train = 4
            self.n_valid = 4
            self.n_test = 4
            self.train_display_interval = 1 
            self.valid_display_interval = 1 
            self.test_display_interval = 1 
            self.train_max_iters = 100
            self.valid_max_iters = self.n_valid/ self.valid_batch_size
            self.test_max_iters  = self.n_test / self.test_batch_size
        elif dataset == 'msrp_dpool':
            self.train_data_file = '/home/wsx/data/msrp_dpool/train'
            self.valid_data_file = '/home/wsx/data/msrp_dpool/valid'
            self.test_data_file  = '/home/wsx/data/msrp_dpool/test'
            
            self.feat_size = 25

            self.dp_rate = 0.5
            self.num_class = 2
            self.batch_size = 50
            self.train_batch_size = 50
            self.valid_batch_size = 50
            self.test_batch_size  = 50 
            self.n_train = 7152
            self.n_valid = 500
            self.n_test = 1725
            self.train_display_interval = 1
            self.valid_display_interval = 100
            self.test_display_interval  = 100
            self.train_max_iters = 5000
            self.valid_max_iters = self.n_valid/ self.valid_batch_size
            self.test_max_iters  = self.n_test / self.test_batch_size
        else:
            assert False

