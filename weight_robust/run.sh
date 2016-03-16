#! bin/sh

BASE_DIR=/home/pangliang/matching/data/robust/origin

./trec_eval.9.0/trec_eval trecRelGTFile_1conv.14000.txt trecResFile_TF.txt -m all_trec
# ./trec_eval.9.0/trec_eval trecRelGTFile_1conv.14000.txt trecResFile_TF_log.txt -m all_trec
# ./trec_eval.9.0/trec_eval trecRelGTFile_1conv.14000.txt trecResFile_TFIDF_log.txt -m all_trec
# ./trec_eval.9.0/trec_eval trecRelGTFile_1conv.14000.txt trecResFile_TFIDF_norm.txt -m all_trec
# ./trec_eval.9.0/trec_eval trecRelGTFile_1conv.14000.txt trecResFile_TFIDF.txt -m all_trec
# ./trec_eval.9.0/trec_eval trecRelGTFile_1conv.14000.txt trecResFile_BM25.txt -m all_trec
#./trec_eval.9.0/trec_eval trecRelGTFile_1conv.14000.txt trecResFile_1conv.14000.txt -m all_trec
#./trec_eval.9.0/trec_eval $BASE_DIR/robust04.qrels $BASE_DIR/rob04.titles.galago.2k.out -m all_trec

