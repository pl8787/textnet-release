#ifndef TEXTNET_LAYER_IMPL_INL_HPP_
#define TEXTNET_LAYER_IMPL_INL_HPP_

#include "./layer.h"
#include "./common/activation_layer-inl.hpp"
#include "./common/convolution_layer-inl.hpp"
#include "./common/fullc_layer-inl.hpp"
#include "./common/tensor_fullc_layer-inl.hpp"
#include "./common/pooling_layer-inl.hpp"
#include "./common/embedding_layer-inl.hpp"
#include "./common/one_hot_layer-inl.hpp"
#include "./common/cross_layer-inl.hpp"
#include "./common/split_layer-inl.hpp"
#include "./common/conv_result_transform_layer-inl.hpp"
#include "./common/conv_lstm_split_layer-inl.hpp"
#include "./common/dropout_layer-inl.hpp"
#include "./common/match_layer-inl.hpp"
#include "./common/match_tensor_layer-inl.hpp"
#include "./common/match_tensor_fact_layer-inl.hpp"
#include "./common/match_weighted_dot_layer-inl.hpp"
#include "./common/match_multi_layer-inl.hpp"
#include "./common/batch_combine_layer-inl.hpp"
#include "./common/batch_select_layer-inl.hpp"
#include "./common/batch_split_layer-inl.hpp"
#include "./common/batch_concat_layer-inl.hpp"
#include "./common/batch_duplicate_layer-inl.hpp"
#include "./common/lstm_layer-inl.hpp"
#include "./common/lstm_d2_layer-inl.hpp"
#include "./common/gru_layer-inl.hpp"
#include "./common/lstm_autoencoder_layer-inl.hpp"
#include "./input/lstm_autoencoder_input_layer-inl.hpp"
#include "./common/product_layer-inl.hpp"
#include "./common/sum_layer-inl.hpp"
#include "./common/recurrent_layer-inl.hpp"
#include "./common/max_recurrent_layer-inl.hpp"
#include "./common/diag_recurrent_layer-inl.hpp"
#include "./common/convolutional_lstm_layer-inl.hpp"
#include "./common/whole_pooling_layer-inl.hpp"
#include "./common/topk_pooling_layer-inl.hpp"
#include "./common/concat_layer-inl.hpp"
#include "./common/gate_layer-inl.hpp"
#include "./common/gate_alldim_layer-inl.hpp"
#include "./common/softmax_func_layer-inl.hpp"
#include "./common/softmax_func_var_len_layer-inl.hpp"
#include "./common/sequcence_dim_reduction_layer-inl.hpp"
#include "./common/gating_layer-inl.hpp"
#include "./common/dynamic_pooling_layer-inl.hpp"
#include "./common/dynamic_k_max_pooling_layer-inl.hpp"
#include "./common/duplicate4lstm_layer-inl.hpp"
#include "./common/swap_axis_layer-inl.hpp"
#include "./common/flatten_layer-inl.hpp"
#include "./common/lr2softmax_layer-inl.hpp"
#include "./common/pos_pred_rep_layer-inl.hpp"
#include "./common/nbp_gen_lstm_input_layer-inl.hpp"
#include "./common/phrase_ave_rep_layer-inl.hpp"
#include "./common/match_topk_pooling_layer-inl.hpp"
#include "./common/select_sub_rep_by_token_layer-inl.hpp"
#include "./input/textdata_layer-inl.hpp"
#include "./input/next_basket_data_layer-inl.hpp"
#include "./input/sequence_classification_data_layer-inl.hpp"
#include "./input/negative_sample_layer-inl.hpp"
#include "./input/lm_input_layer-inl.hpp"
#include "./input/label_feat_value_layer-inl.hpp"
#include "./input/match_phrase_rep_layer-inl.hpp"
#include "./input/pair_textdata_layer-inl.hpp"
#include "./input/list_textdata_layer-inl.hpp"
#include "./input/qa_textdata_layer-inl.hpp"
#include "./input/word_rep_input_layer-inl.hpp"
#include "./loss/hingeloss_layer-inl.hpp"
#include "./loss/cross_entropy_loss_layer-inl.hpp"
#include "./loss/pairhingeloss_layer-inl.hpp"
#include "./loss/softmax_layer-inl.hpp"
#include "./loss/accuracy_layer-inl.hpp"
#include "./loss/negative_sample_loss_layer-inl.hpp"
#include "./loss/word_class_softmax_loss_layer-inl.hpp"
#include "./loss/lm_softmax_loss_layer-inl.hpp"
#include "./loss/listwise_measure_layer-inl.hpp"
#include "./loss/euclid_distance_loss_layer-inl.hpp"

namespace textnet {
namespace layer {
template<typename xpu>
Layer<xpu>* CreateLayer_(LayerType type) {
  switch(type) {
    case kSigmoid: return new ActivationLayer<xpu, op::sigmoid, op::sigmoid_grad>(type);
    case kTanh: return new ActivationLayer<xpu, op::tanh, op::tanh_grad>(type);
    case kRectifiedLinear: return new ActivationLayer<xpu, op::relu, op::relu_grad>(type);
    case kConv: return new ConvolutionLayer<xpu>(type);
    case kFullConnect: return new FullConnectLayer<xpu>(type);
    case kTensorFullConnect: return new TensorFullConnectLayer<xpu>(type);
    case kMaxPooling: return new PoolingLayer<mshadow::red::maximum, xpu>(type);
    case kAvgPooling: return new PoolingLayer<mshadow::red::sum, xpu>(type);
    case kWholePooling: return new WholePoolingLayer<xpu>(type);
    case kTopkPooling: return new TopkPoolingLayer<xpu>(type);
    case kPosPredRep: return new PosPredRepLayer<xpu>(type);
    case kConcat: return new ConcatLayer<xpu>(type);
    case kEmbedding: return new EmbeddingLayer<xpu>(type);
    case kOneHot: return new OneHotLayer<xpu>(type);
    case kCross: return new CrossLayer<xpu>(type);
    case kSplit: return new SplitLayer<xpu>(type);
    case kDup4lstm: return new Duplicate4lstmLayer<xpu>(type);
    case kConvResultTransform: return new ConvResultTransformLayer<xpu>(type);
    case kConvLstmSplit: return new SplitLayer<xpu>(type);
    case kDropout: return new DropoutLayer<xpu>(type);
    case kDynamicPooling: return new DynamicPoolingLayer<xpu>(type);
    case kDynamicKMaxPooling: return new DynamicKMaxPoolingLayer<xpu>(type);
    case kLstm: return new LstmLayer<xpu>(type);
    case kLstmD2: return new LstmD2Layer<xpu>(type);
    case kLstmAutoencoder: return new LstmAutoencoderLayer<xpu>(type);
    case kLstmAutoencoderInput: return new LstmAutoencoderInputLayer<xpu>(type);
    case kNbpGenLstmInput: return new NbpGenLstmInputLayer<xpu>(type);
    case kPhraseAveRep: return new PhraseAveRepLayer<xpu>(type);
    case kProduct: return new ProductLayer<xpu>(type);
    case kRecurrent: return new RecurrentLayer<xpu>(type);
    case kMaxRecurrent: return new MaxRecurrentLayer<xpu>(type);
    case kDiagRecurrent: return new DiagRecurrentLayer<xpu>(type);
    case kSequenceDimReduction: return new SequenceDimReductionLayer<xpu>(type);
    case kConvolutionalLstm: return new ConvolutionalLstmLayer<xpu>(type);
    case kGate: return new GateLayer<xpu>(type);
    case kLr2softmax: return new Lr2softmaxLayer<xpu>(type);
    case kGateAlldim: return new GateAlldimLayer<xpu>(type);
	case kGating: return new GatingLayer<xpu>(type);
    case kHingeLoss: return new HingeLossLayer<xpu>(type);
    case kPairHingeLoss: return new PairHingeLossLayer<xpu>(type);
    case kCrossEntropyLoss: return new CrossEntropyLossLayer<xpu>(type);
    case kTextData: return new TextDataLayer<xpu>(type);
    case kNextBasketData: return new NextBasketDataLayer<xpu>(type);
    case kSequenceClassificationData: return new SequenceClassificationDataLayer<xpu>(type);
    case kNegativeSample: return new NegativeSampleLayer<xpu>(type);
    case kSoftmax: return new SoftmaxLayer<xpu>(type);
    case kSoftmaxFunc: return new SoftmaxFuncLayer<xpu>(type);
    case kSoftmaxFuncVarLen: return new SoftmaxFuncVarLenLayer<xpu>(type);
    case kNegativeSampleLoss: return new NegativeSampleLossLayer<xpu>(type);
    case kWordClassSoftmaxLoss: return new WordClassSoftmaxLossLayer<xpu>(type);
    case kLmSoftmaxLoss: return new LmSoftmaxLossLayer<xpu>(type);
    case kLmInput: return new LmInputLayer<xpu>(type);
    case kLabelFeatValue: return new LabelFeatValueLayer<xpu>(type);
    case kSumByAxis: return new SumLayer<xpu>(type);
    case kAccuracy: return new AccuracyLayer<xpu>(type);
    case kMatch: return new MatchLayer<xpu>(type);
    case kMatchTensor: return new MatchTensorLayer<xpu>(type);
    case kMatchTensorFact: return new MatchTensorFactLayer<xpu>(type);
    case kMatchWeightedDot: return new MatchWeightedDotLayer<xpu>(type);
	case kMatchMulti: return new MatchMultiLayer<xpu>(type);
	case kMatchTopKPooling: return new MatchTopKPoolingLayer<xpu>(type);
    case kBatchCombine: return new BatchCombineLayer<xpu>(type);
    case kBatchSelect: return new BatchSelectLayer<xpu>(type);
    case kBatchSplit: return new BatchSplitLayer<xpu>(type);
    case kBatchConcat: return new BatchConcatLayer<xpu>(type);
    case kBatchDuplicate: return new BatchDuplicateLayer<xpu>(type);
    case kSwapAxis: return new SwapAxisLayer<xpu>(type);
    case kFlatten: return new FlattenLayer<xpu>(type);
    case kMatchPhraseRep: return new MatchPhraseRepLayer<xpu>(type);
    case kGru: return new GruLayer<xpu>(type);
	case kPairTextData: return new PairTextDataLayer<xpu>(type);
	case kListTextData: return new ListTextDataLayer<xpu>(type);
	case kListwiseMeasure: return new ListwiseMeasureLayer<xpu>(type);
	case kQATextData: return new QATextDataLayer<xpu>(type);
    case kSelectSubRepByToken: return new SelectSubRepByTokenLayer<xpu>(type);
    case kWordRepInput: return new WordRepInputLayer<xpu>(type);
    case kEuclidDistanceLoss: return new EuclidDistanceLossLayer<xpu>(type);
    default: utils::Error("unknown layer type id : \"%d\"", type); return NULL;
  }
}

}  // namespace layer
}  // namespace textnet
#endif
