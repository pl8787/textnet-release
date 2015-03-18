#ifndef TEXTNET_LAYER_IMPL_INL_HPP_
#define TEXTNET_LAYER_IMPL_INL_HPP_

#include "./layer.h"
#include "./common/activation_layer-inl.hpp"
#include "./common/convolution_layer-inl.hpp"
#include "./common/fullc_layer-inl.hpp"
#include "./common/pooling_layer-inl.hpp"
//#include "./loss/softmax_layer-inl.hpp"

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
    case kMaxPooling: return new PoolingLayer<mshadow::red::maximum, xpu>(kMaxPooling);
    case kAvgPooling: return new PoolingLayer<mshadow::red::sum, xpu>(kAvgPooling);
    //case kSoftmax: return new SoftmaxLayer<xpu>(setting);
    default: utils::Error("unknown layer type id : \"%d\"", type); return NULL;
  }
}

}  // namespace layer
}  // namespace textnet
#endif
