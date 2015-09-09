
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

        elif dataset == 'tf':
            self.train_data_file = '/home/wsx/data/nbp/tf.train.lstm'
            self.valid_data_file = '/home/wsx/data/nbp/tf.valid.lstm'
            self.test_data_file  = '/home/wsx/data/nbp/tf.test.lstm'
            
            self.num_item = 7973
            self.num_user = 2265
            self.max_session_len = 105
            self.max_context_len = 10

            self.dp_rate = 0.0
            self.d_user_rep = 30
            self.d_item_rep = 30

            self.batch_size = 1
            self.train_batch_size = 1 
            self.valid_batch_size = 1 
            self.test_batch_size  = 1
            self.n_train = 30747
            self.n_valid = 2265
            self.n_test = 2265
            self.train_display_interval = 1 
            self.valid_display_interval = 10000
            self.test_display_interval  = 10000 
            self.train_max_iters = 300000
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
            # self.vocab_size = 15586
            self.vocab_size = 50000

            self.dp_rate = 0.0
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
        elif dataset == 'qa_top10':
            self.train_data_file = '/home/wsx/data/qa_top10/qa.neg.10.50.train'
            self.valid_data_file = '/home/wsx/data/qa_top10/qa.neg.10.50.valid'
            self.test_data_file  = '/home/wsx/data/qa_top10/qa.neg.10.50.test'
            self.embedding_file  = '/home/wsx/data/qa_top10/qa_embed'
            # self.update_indication_file = '/home/wsx/data/msrp/wikicorp_num_50_msr_ind.txt'
            self.max_doc_len = 50
            self.min_doc_len = 5 
            # self.vocab_size = 15586
            # self.vocab_size = 219071
            self.vocab_size = 130242

            self.dp_rate = 0.0
            self.num_class = 2
            self.d_word_rep = 50
            self.batch_size = 128
            self.train_batch_size = 128
            self.valid_batch_size = 128
            self.test_batch_size  = 128
            # self.n_train = 1082851
            # self.n_valid = 135355
            # self.n_test  = 135355
            self.train_display_interval = 10
            self.valid_display_interval = 10000000
            self.test_display_interval  = 1000
            self.train_max_iters = 20000
            self.valid_max_iters =6056
            self.test_max_iters  =6056
        elif dataset == 'qa_top300':
            self.train_data_file = '/home/wsx/data/qa_top300/qa.neg.10.50.train'
            self.valid_data_file = '/home/wsx/data/qa_top300/qa.neg.10.50.valid'
            self.test_data_file  = '/home/wsx/data/qa_top300/qa.neg.10.50.test'
            self.embedding_file  = '/home/wsx/data/qa_top300/qa_embed'
            # self.update_indication_file = '/home/wsx/data/msrp/wikicorp_num_50_msr_ind.txt'
            self.max_doc_len = 50
            self.min_doc_len = 5 
            # self.vocab_size = 15586
            # self.vocab_size = 219071
            self.vocab_size = 130242

            self.dp_rate = 0.0
            self.num_class = 2
            self.d_word_rep = 50
            self.batch_size = 128
            self.train_batch_size = 128
            self.valid_batch_size = 128
            self.test_batch_size  = 128
            # self.n_train = 1082851
            # self.n_valid = 135355
            # self.n_test  = 135355
            self.train_display_interval = 10
            self.valid_display_interval = 10000000
            self.test_display_interval  = 1000
            self.train_max_iters = 20000
            self.valid_max_iters =6056
            self.test_max_iters  =6056
        elif dataset == 'qa_top1k_4':
            self.train_data_file = '/home/wsx/data/qa_top1k_4/qa.neg.4.train'
            self.valid_data_file = '/home/wsx/data/qa_top1k_4/qa.neg.4.valid'
            self.test_data_file  = '/home/wsx/data/qa_top1k_4/qa.neg.4.test'
            self.embedding_file  = '/home/wsx/data/qa_top1k_4/qa_embed'
            # self.update_indication_file = '/home/wsx/data/msrp/wikicorp_num_50_msr_ind.txt'
            self.max_doc_len = 50
            self.min_doc_len = 5 
            # self.vocab_size = 15586
            # self.vocab_size = 219071
            self.vocab_size = 130242

            self.dp_rate = 0.0
            self.num_class = 2
            self.d_word_rep = 50
            self.batch_size = 128
            self.train_batch_size = 128
            self.valid_batch_size = 128
            self.test_batch_size  = 128
            # self.n_train = 1082851
            # self.n_valid = 135355
            # self.n_test  = 135355
            self.train_display_interval = 10
            self.valid_display_interval = 10000000
            self.test_display_interval  = 1000
            self.train_max_iters = 20000
            self.valid_max_iters =6056
            self.test_max_iters  =6056
        elif dataset == 'qa_top1k':
            self.train_data_file = '/home/wsx/data/qa_top1k/qa.neg.10.50.train'
            self.valid_data_file = '/home/wsx/data/qa_top1k/qa.neg.10.50.valid'
            self.test_data_file  = '/home/wsx/data/qa_top1k/qa.neg.10.50.test'
            self.embedding_file  = '/home/wsx/data/qa_top1k/qa_embed'
            # self.update_indication_file = '/home/wsx/data/msrp/wikicorp_num_50_msr_ind.txt'
            self.max_doc_len = 50
            self.min_doc_len = 5 
            # self.vocab_size = 15586
            # self.vocab_size = 219071
            self.vocab_size = 130242

            self.dp_rate = 0.0
            self.num_class = 2
            self.d_word_rep = 50
            self.batch_size = 128
            self.train_batch_size = 128
            self.valid_batch_size = 128
            self.test_batch_size  = 128
            # self.n_train = 1082851
            # self.n_valid = 135355
            # self.n_test  = 135355
            self.train_display_interval = 10
            self.valid_display_interval = 10000000
            self.test_display_interval  = 1000
            self.train_max_iters = 20000
            self.valid_max_iters =6056
            self.test_max_iters  =6056

        elif dataset == 'sentence':
            self.train_data_file = '/home/wsx/data/sentence/train'
            self.valid_data_file = '/home/wsx/data/sentence/test'
            self.test_data_file  = '/home/wsx/data/sentence/test'
            self.embedding_file  = '/home/wsx/data/sentence/sentence_embed'
            # self.update_indication_file = '/home/wsx/data/msrp/wikicorp_num_50_msr_ind.txt'
            self.max_doc_len = 28
            self.min_doc_len = 4 
            # self.vocab_size = 15586
            # self.vocab_size = 219071
            self.vocab_size = 127889

            self.dp_rate = 0.0
            self.num_class = 2
            self.d_word_rep = 50
            self.batch_size = 128
            self.train_batch_size = 128
            self.valid_batch_size = 128
            self.test_batch_size  = 128
            # self.n_train = 1082851
            # self.n_valid = 135355
            # self.n_test  = 135355
            self.train_display_interval = 10
            self.valid_display_interval = 10000000
            self.test_display_interval  = 4000
            self.train_max_iters = 40001
            self.valid_max_iters = 50000
            self.test_max_iters  = 50000
        elif dataset == 'qa_50':
            self.train_data_file = '/home/wsx/data/qa_50/qa.neg.10.50.train'
            self.valid_data_file = '/home/wsx/data/qa_50/qa.neg.10.50.valid'
            self.test_data_file  = '/home/wsx/data/qa_50/qa.neg.10.50.test'
            self.embedding_file  = '/home/wsx/data/qa_50/qa_embed'
            # self.update_indication_file = '/home/wsx/data/msrp/wikicorp_num_50_msr_ind.txt'
            self.max_doc_len = 50
            self.min_doc_len = 5 
            # self.vocab_size = 15586
            # self.vocab_size = 219071
            self.vocab_size = 130242

            self.dp_rate = 0.0
            self.num_class = 2
            self.d_word_rep = 50
            self.batch_size = 128
            self.train_batch_size = 128
            self.valid_batch_size = 128
            self.test_batch_size  = 128
            # self.n_train = 1082851
            # self.n_valid = 135355
            # self.n_test  = 135355
            self.train_display_interval = 10
            self.valid_display_interval = 10000000
            self.test_display_interval  = 2000
            self.train_max_iters = 20001
            self.valid_max_iters =6057
            self.test_max_iters  =6057

        elif dataset == 'qa':
            self.train_data_file = '/home/wsx/data/qa/qa.neg.xrear10.3.32.train.dat'
            self.valid_data_file = '/home/wsx/data/qa/qa.neg.xrear10.3.32.valid.dat'
            self.test_data_file  = '/home/wsx/data/qa/qa.neg.xrear10.3.32.test.dat'
            self.embedding_file  = '/home/wsx/data/qa/qa_embed'
            # self.update_indication_file = '/home/wsx/data/msrp/wikicorp_num_50_msr_ind.txt'
            self.max_doc_len = 33
            self.min_doc_len = 1 
            # self.vocab_size = 15586
            # self.vocab_size = 219071
            self.vocab_size = 120750

            self.dp_rate = 0.0
            self.num_class = 2
            self.d_word_rep = 50
            self.batch_size = 128
            self.train_batch_size = 128
            self.valid_batch_size = 128
            self.test_batch_size  = 128
            # self.n_train = 1082851
            # self.n_valid = 135355
            # self.n_test  = 135355
            self.train_display_interval = 10
            self.valid_display_interval = 10000000
            self.test_display_interval  = 1000
            self.train_max_iters = 40000
            self.valid_max_iters =12303
            self.test_max_iters  =12303
        elif dataset == 'qa_candi':
            self.train_data_file = '/home/wsx/data/qa/qa.xmore10.32.train.dat'
            self.valid_data_file = '/home/wsx/data/qa/qa.xmore10.32.valid.dat'
            self.test_data_file  = '/home/wsx/data/qa/qa.xmore10.32.test.dat'
            self.embedding_file  = '/home/wsx/data/qa/qa_embed'
            # self.update_indication_file = '/home/wsx/data/msrp/wikicorp_num_50_msr_ind.txt'
            self.max_doc_len = 33
            self.min_doc_len = 5 
            # self.vocab_size = 15586
            self.vocab_size = 219071

            self.dp_rate = 0.0
            self.num_class = 2
            self.d_word_rep = 50
            self.batch_size = 128
            self.train_batch_size = 128
            self.valid_batch_size = 128
            self.test_batch_size  = 128
            self.n_train = 1082851
            self.n_valid = 135355
            self.n_test  = 135355
            self.train_display_interval = 10
            self.valid_display_interval = 500
            self.test_display_interval  = 500
            self.train_max_iters = 20000
            self.valid_max_iters =12305
            self.test_max_iters  =12305
        elif dataset == 'qa_balance':
            self.train_data_file = '/home/wsx/data/qa/qa.neg.xmore10.32.train.dat.balance'
            self.valid_data_file = '/home/wsx/data/qa/qa.neg.xmore10.32.valid.dat.balance'
            self.test_data_file  = '/home/wsx/data/qa/qa.neg.xmore10.32.test.dat.balance'
            self.embedding_file  = '/home/wsx/data/qa/qa_embed'
            # self.update_indication_file = '/home/wsx/data/msrp/wikicorp_num_50_msr_ind.txt'
            self.max_doc_len = 33
            self.min_doc_len = 5 
            # self.vocab_size = 15586
            self.vocab_size = 219071

            self.dp_rate = 0.0
            self.num_class = 2
            self.d_word_rep = 50
            self.batch_size = 128
            self.train_batch_size = 128
            self.valid_batch_size = 128
            self.test_batch_size  = 128
            self.n_train = 196882
            self.n_valid = 24610
            self.n_test  = 24610
            self.train_display_interval = 10
            self.valid_display_interval = 500
            self.test_display_interval  = 500
            self.train_max_iters = 20000
            self.valid_max_iters = self.n_valid / self.valid_batch_size
            self.test_max_iters  = self.n_test  / self.test_batch_size
        elif dataset == 'msrp_seq':
            self.train_data_file = '/home/wsx/data/msrp/train.seq'
            self.valid_data_file = '/home/wsx/data/msrp/valid.seq'
            self.test_data_file  = '/home/wsx/data/msrp/test.seq'
            # self.embedding_file  = '/home/wsx/data/msrp/msrp.embed'
            # self.update_indication_file = '/home/wsx/data/msrp/wikicorp_num_50_msr_ind.txt'
            
            self.max_doc_len = 33
            self.min_doc_len = 5 
            # self.vocab_size = 15586
            # self.vocab_size = 50000

            self.dp_rate = 0.0
            self.num_class = 2
            # self.d_word_rep = 50
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

        elif dataset == 'nyt':
            self.data_dir = '/home/wsx/data/nyt/'
            self.train_data_file = self.data_dir + 'nyt.wid.train.with_msrp'
            self.valid_data_file = self.data_dir + 'nyt.wid.valid'
            self.test_data_file  = self.data_dir + 'msrp.sentence.valid'
            # self.embedding_file  = self.data_dir + 'wiki.embed'
            # self.update_indication_file = self.data_dir + 'wiki.ind'
            # self.word_class_file = self.data_dir + 'id2class'
            # self.word_freq_file = self.data_dir + 'word_freq'
            
            self.max_doc_len = 60
            self.min_doc_len = 0
            self.vocab_size = 45844 # without orc_unknown

            self.dp_rate = 0.
            self.d_word_rep = 1000
            self.batch_size = 32
            self.train_batch_size = 32
            self.valid_batch_size = 32
            self.test_batch_size  = 32
            self.n_train = 111456 
            self.n_valid = 10000
            self.n_test = 1000
            self.train_display_interval = 1
            self.valid_display_interval = 500 
            self.test_display_interval  = 500 
            self.train_max_iters = (self.n_train/self.train_batch_size) * 5
            self.valid_max_iters = (self.n_valid/10)/self.valid_batch_size
            self.test_max_iters  = (self.n_test)/self.test_batch_size
        elif dataset == 'wiki':
            self.data_dir = '/home/wsx/data/wiki/'
            self.train_data_file = self.data_dir + 'wiki.train.with_msrp'
            self.valid_data_file = self.data_dir + 'wiki.valid'
            self.test_data_file  = self.data_dir + 'msrp.sentence.valid'
            # self.embedding_file  = self.data_dir + 'wiki.embed'
            # self.update_indication_file = self.data_dir + 'wiki.ind'
            self.word_class_file = self.data_dir + 'id2class'
            # self.word_freq_file = self.data_dir + 'word_freq'
            
            self.max_doc_len = 50
            self.min_doc_len = 5
            self.vocab_size = 177859 # without orc_unknown

            self.dp_rate = 0.
            self.d_word_rep = 2
            self.batch_size = 10
            self.train_batch_size = 10
            self.valid_batch_size = 10
            self.test_batch_size  = 10
            self.n_train = 924735
            self.n_valid = 94802
            self.n_test = 1000
            self.train_display_interval = 1
            self.valid_display_interval = 1000 
            self.test_display_interval  = 1000 
            self.train_max_iters = (self.n_train/self.train_batch_size) * 5
            self.valid_max_iters = (self.n_valid/50)/self.valid_batch_size
            self.test_max_iters  = self.n_test /self.test_batch_size
            self.valid_display_interval = 50
            self.test_display_interval  = 50 
            self.train_max_iters = 10000
            self.valid_max_iters = self.n_valid/ self.valid_batch_size
            self.test_max_iters  = self.n_test / self.test_batch_size

        elif dataset == 'webscope':
            self.train_data_file = '/home/pangliang/matching/data/webscope/qa_instances.train.dat'
            self.valid_data_file = '/home/pangliang/matching/data/webscope/qa_instances.valid.dat'
            self.test_data_file  = '/home/pangliang/matching/data/webscope/qa_instances.test.dat'
            self.embedding_file  = ''
            self.update_indication_file = ''
            
            self.max_doc_len = 32
            self.min_doc_len = 5 
            self.vocab_size = 214555

            self.dp_rate = 0.5
            self.num_class = 2
            self.d_word_rep = 50
            self.batch_size = 128
            self.train_batch_size = 128
            self.valid_batch_size = 128
            self.test_batch_size  = 128
            self.n_train = 114103
            self.n_valid = 14262 
            self.n_test = 14262
            self.train_display_interval = 1 
            self.valid_display_interval = 200
            self.test_display_interval  = 200 
            self.train_max_iters = 100000
            self.valid_max_iters = self.n_valid/ self.valid_batch_size
            self.test_max_iters  = self.n_test / self.test_batch_size

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
            self.data_dir = '/home/wsx/data/test/test_lm/'
            self.train_data_file = self.data_dir + 'train.txt'
            self.valid_data_file = self.data_dir + 'train.txt'
            self.test_data_file  = self.data_dir + 'train.txt'
            self.word_class_file = self.data_dir + 'id2class'
            self.word_freq_file = self.data_dir + 'word_freq'
            
            self.max_doc_len = 6
            self.min_doc_len = 0
            self.vocab_size = 8 # without orc_unknown

            self.dp_rate = 0.
            self.d_word_rep = 5
            self.batch_size = 2
            self.train_batch_size = 2
            self.valid_batch_size = 2
            self.test_batch_size  = 2
            self.n_train = 4
            self.n_valid = 4
            self.n_test = 4 
            self.train_display_interval = 1
            self.valid_display_interval = 1 
            self.test_display_interval  = 1 
            self.train_max_iters = (self.n_train/self.train_batch_size) * 5
            self.valid_max_iters = (self.n_valid/5)/self.valid_batch_size
            self.test_max_iters  = self.n_test /self.test_batch_size
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

