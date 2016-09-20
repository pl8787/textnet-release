#ifndef TEXTNET_LAYER_IMPL_INL_HPP_
#define TEXTNET_LAYER_IMPL_INL_HPP_

#include "./layer.h"
#include "./common/activation_layer-inl.hpp"
#include "./common/xelu_layer-inl.hpp"
#include "./common/elu_layer-inl.hpp"
#include "./common/append_feature_layer-inl.hpp"
#include "./common/convolution_layer-inl.hpp"
#include "./common/convolution_var_layer-inl.hpp"
#include "./common/convolution_param_layer-inl.hpp"
#include "./common/gen_kernel_layer-inl.hpp"
#include "./common/fullc_layer-inl.hpp"
#include "./common/tensor_fullc_layer-inl.hpp"
#include "./common/pooling_layer-inl.hpp"
#include "./common/pooling_var_layer-inl.hpp"
#include "./common/pad_layer-inl.hpp"
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
#include "./common/batch_norm_layer-inl.hpp"
#include "./common/batch_max_layer-inl.hpp"
#include "./common/channel_duplicate_layer-inl.hpp"
#include "./common/lstm_layer-inl.hpp"
#include "./common/lstm_d2_layer-inl.hpp"
#include "./common/lstm_d2_optimize_layer-inl.hpp"
#include "./common/gru_d2_layer-inl.hpp"
#include "./common/gru_d2_one_gate_layer-inl.hpp"
// #include "./common/gru_d2_optimize_layer-inl.hpp"
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
#include "./common/whole_pooling_2d_layer-inl.hpp"
#include "./common/blstm_layer-inl.hpp"
#include "./common/bgru_d2_layer-inl.hpp"

#ifdef CPU_ONLY
#include "./common/gate_whole_pooling_layer-inl.hpp"
#include "./common/gate_whole_pooling_d2_layer-inl.hpp"
#include "./common/softmax_func_var_len_layer-inl.hpp"
#endif

#include "./common/gate_dynamic_pooling_d2_layer-inl.hpp"
#include "./common/topk_pooling_layer-inl.hpp"
#include "./common/concat_layer-inl.hpp"
#include "./common/gate_layer-inl.hpp"
#include "./common/gate_alldim_layer-inl.hpp"
#include "./common/softmax_func_layer-inl.hpp"
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
#include "./common/local_layer-inl.hpp"
#include "./common/local_factor_layer-inl.hpp"
#include "./common/gaussian_mask_layer-inl.hpp"
#include "./common/memory_attention_in_layer-inl.hpp"
#include "./common/memory_attention_out_layer-inl.hpp"
#include "./common/augmentation_layer-inl.hpp"
#include "./common/element_op_layer-inl.hpp"
#include "./common/parameter_layer-inl.hpp"
#include "./common/fill_curve_xy2d_layer-inl.hpp"
#include "./common/fill_curve_d2xy_layer-inl.hpp"
#include "./common/length_trans_layer-inl.hpp"
#include "./common/length_fill_layer-inl.hpp"
#include "./common/axis_split_layer-inl.hpp"
#include "./common/key_snip_layer-inl.hpp"
#include "./common/reshape_layer-inl.hpp"
#include "./common/merge_2_window_data_layer-inl.hpp"
#include "./input/textdata_layer-inl.hpp"
#include "./input/lcs_toy_data_layer-inl.hpp"
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
#include "./input/map_textdata_layer-inl.hpp"
#include "./input/map_2_textdata_layer-inl.hpp"
#include "./input/map_3_textdata_layer-inl.hpp"
#include "./input/map_2_window_textdata_layer-inl.hpp"
#include "./input/image_layer-inl.hpp"
#include "./input/memory_global_layer-inl.hpp"
#include "./loss/hingeloss_layer-inl.hpp"
#include "./loss/cross_entropy_loss_layer-inl.hpp"
#include "./loss/pairhingeloss_layer-inl.hpp"
#include "./loss/pair_weighted_hinge_loss_layer-inl.hpp"
#include "./loss/listhingeloss_layer-inl.hpp"
#include "./loss/softmax_layer-inl.hpp"
#include "./loss/accuracy_layer-inl.hpp"
#include "./loss/negative_sample_loss_layer-inl.hpp"
#include "./loss/word_class_softmax_loss_layer-inl.hpp"
#include "./loss/lm_softmax_loss_layer-inl.hpp"
#include "./loss/listwise_measure_layer-inl.hpp"
#include "./loss/euclid_distance_loss_layer-inl.hpp"
#include "./loss/logistic_layer-inl.hpp"
#include "./loss/activation_norm_loss_layer-inl.hpp"
#include "./common/match_histogram_layer-inl.hpp"
#include "./common/sort_axis_layer-inl.hpp"

namespace textnet {
namespace layer {
template<typename xpu>
Layer<xpu>* CreateLayer_(LayerType type) {
  switch(type) {
    case kSigmoid: return new ActivationLayer<xpu, op::sigmoid, op::sigmoid_grad>(type);
    case kTanh: return new ActivationLayer<xpu, op::tanh, op::tanh_grad>(type);
    case kRectifiedLinear: return new ActivationLayer<xpu, op::relu, op::relu_grad>(type);
    case kConv: return new ConvolutionLayer<xpu>(type);
    case kConvVar: return new ConvolutionVarLayer<xpu>(type);
    case kConvParam: return new ConvolutionParamLayer<xpu>(type);
    case kPoolingVar: return new PoolingVarLayer<xpu>(type);
    case kPad: return new PadLayer<xpu>(type);
    case kFullConnect: return new FullConnectLayer<xpu>(type);
    case kTensorFullConnect: return new TensorFullConnectLayer<xpu>(type);
    case kMaxPooling: return new PoolingLayer<mshadow::red::maximum, xpu>(type);
    case kAvgPooling: return new PoolingLayer<mshadow::red::sum, xpu>(type);
    case kSumPooling: return new PoolingLayer<mshadow::red::sum, xpu>(type);
#ifdef CPU_ONLY
    case kGateWholePooling: return new GateWholePoolingLayer<xpu>(type);
    case kGateWholePoolingD2: return new GateWholePoolingD2Layer<xpu>(type);
    case kSoftmaxFuncVarLen: return new SoftmaxFuncVarLenLayer<xpu>(type);
#endif
    case kGateDynamicPoolingD2: return new GateDynamicPoolingD2Layer<xpu>(type);
    case kWholePooling: return new WholePoolingLayer<xpu>(type);
    case kWholePooling2d: return new WholePooling2dLayer<xpu>(type);
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
    case kLstmD2Optimize: return new LstmD2OptimizeLayer<xpu>(type);
    case kGruD2: return new GruD2Layer<xpu>(type);
    case kGruD2OneGate: return new GruD2OneGateLayer<xpu>(type);
    // case kGruD2Optimize: return new GruD2OptimizeLayer<xpu>(type);
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
    case kPairWeightedHingeLoss: return new PairWeightedHingeLossLayer<xpu>(type);
    case kListHingeLoss: return new ListHingeLossLayer<xpu>(type);
    case kCrossEntropyLoss: return new CrossEntropyLossLayer<xpu>(type);
    case kTextData: return new TextDataLayer<xpu>(type);
    case kLcsToyData: return new LcsToyDataLayer<xpu>(type);
    case kNextBasketData: return new NextBasketDataLayer<xpu>(type);
    case kSequenceClassificationData: return new SequenceClassificationDataLayer<xpu>(type);
    case kNegativeSample: return new NegativeSampleLayer<xpu>(type);
    case kSoftmax: return new SoftmaxLayer<xpu>(type);
    case kSoftmaxFunc: return new SoftmaxFuncLayer<xpu>(type);
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
    case kBatchMax: return new BatchMaxLayer<xpu>(type);
    case kChannelDuplicate: return new ChannelDuplicateLayer<xpu>(type);
    case kSwapAxis: return new SwapAxisLayer<xpu>(type);
    case kFlatten: return new FlattenLayer<xpu>(type);
    case kMatchPhraseRep: return new MatchPhraseRepLayer<xpu>(type);
    case kGru: return new GruLayer<xpu>(type);
    case kPairTextData: return new PairTextDataLayer<xpu>(type);
    case kListTextData: return new ListTextDataLayer<xpu>(type);
    case kListwiseMeasure: return new ListwiseMeasureLayer<xpu>(type);
    case kQATextData: return new QATextDataLayer<xpu>(type);
    case kMapTextData: return new MapTextDataLayer<xpu>(type);
    case kMap2TextData: return new Map2TextDataLayer<xpu>(type);
    case kMap3TextData: return new Map3TextDataLayer<xpu>(type);
    case kMap2WindowTextData:    return new Map2WindowTextDataLayer<xpu>(type);
    case kMerge2WindowData:    return new Merge2WindowDataLayer<xpu>(type);
    case kSelectSubRepByToken: return new SelectSubRepByTokenLayer<xpu>(type);
    case kWordRepInput: return new WordRepInputLayer<xpu>(type);
    case kEuclidDistanceLoss: return new EuclidDistanceLossLayer<xpu>(type);
    case kLogistic: return new LogisticLayer<xpu>(type);
    case kActivationNormLoss: return new ActivationNormLossLayer<xpu>(type);
    case kLocal: return new LocalLayer<xpu>(type);
    case kLocalFactor: return new LocalFactorLayer<xpu>(type);
    case kImage: return new ImageLayer<xpu>(type);
    case kGaussianMask: return new GaussianMaskLayer<xpu>(type);
    case kMemoryGlobal: return new MemoryGlobalLayer<xpu>(type);
    case kMemoryAttentionIn: return new MemoryAttentionInLayer<xpu>(type);
    case kMemoryAttentionOut: return new MemoryAttentionOutLayer<xpu>(type);
    case kAugmentation: return new AugmentationLayer<xpu>(type);
    case kElementOp: return new ElementOpLayer<xpu>(type);
    case kParameter: return new ParameterLayer<xpu>(type);
    case kGenKernel: return new GenKernelLayer<xpu>(type);
    case kBatchNorm: return new BatchNormLayer<xpu>(type);
    case kFillCurveXY2D: return new FillCurveXY2DLayer<xpu>(type);
    case kFillCurveD2XY: return new FillCurveD2XYLayer<xpu>(type);
    case kLengthTrans: return new LengthTransLayer<xpu>(type);
    case kLengthFill: return new LengthFillLayer<xpu>(type);
    case kAxisSplit: return new AxisSplitLayer<xpu>(type);
    case kKeySnip: return new KeySnipLayer<xpu>(type);
    case kReshape: return new ReshapeLayer<xpu>(type);
    case kMatchHistogram: return new MatchHistogramLayer<xpu>(type);
    case kSortAxis: return new SortAxisLayer<cpu>(type);
    case kBLstm: return new BLstmLayer<cpu>(type);
    case kBGruD2: return new BGruD2Layer<cpu>(type);
    case kXeLU: return new XeLULayer<xpu>(type);
    case kELU: return new ELULayer<xpu>(type);
    case kAppendFeature: return new AppendFeatureLayer<xpu>(type);
    default: utils::Error("unknown layer type id : \"%d\"", type); return NULL;
  }
}

}  // namespace layer
}  // namespace textnet
#endif
