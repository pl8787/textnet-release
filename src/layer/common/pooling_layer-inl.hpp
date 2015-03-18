#ifndef TEXTNET_LAYER_POOLING_LAYER_INL_HPP_
#define TEXTNET_LAYER_POOLING_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {
  
template<typename Reducer, typename xpu>
class PoolingLayer : public Layer<xpu> {
 public:
  PoolingLayer(LayerType type) { this->layer_type = type; }
  virtual ~PoolingLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top) {
    kernel_x = setting["kernel_x"].i_val;
    kernel_y = setting["kernel_y"].i_val;
    stride = setting["stride"].i_val;
    channel = bottom[0]->data.size(1);
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "PoolingLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "PoolingLayer:top size problem.");
    
    mshadow::Shape<4> shape_in = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_out = mshadow::Shape4(shape_in[0], shape_in[1], 
                   (shape_in[2] - kernel_y) / stride + 1,
                   (shape_in[3] - kernel_x) / stride + 1);
    top[0]->Resize(shape_out);   
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    
    mshadow::Shape<2> pshape = top_data[0][0].shape_;

    if (this->layer_type == kMaxPooling || this->layer_type == kSumPooling) {
      top_data = pool<Reducer>(bottom_data, pshape, kernel_y, kernel_x, stride);
    }else if (this->layer_type == kAvgPooling) {
      top_data = pool<Reducer>(bottom_data, pshape, kernel_y, kernel_x, stride)
          * (1.0f / (kernel_y*kernel_x));
    } else {
      utils::Error("Unknown pooling mode");
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    
    if (this->prop_grad[0]) {
      if (this->layer_type == kMaxPooling || this->layer_type == kSumPooling) {
        bottom_diff = unpool<Reducer>(bottom_data, top_data, top_diff, kernel_y, kernel_x, stride);
      }else if (this->layer_type == kAvgPooling) {
        bottom_diff = unpool<Reducer>(bottom_data, top_data, top_diff, kernel_y, kernel_x, stride)
            * (1.0f / (kernel_y*kernel_x));
      } else {
        utils::Error("Unknown pooling mode");
      }
    }
  }

 protected:
  int kernel_x;
  int kernel_y;
  int stride;
  int channel;
  
};   // class PoolingLayer
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_POOLING_LAYER_INL_HPP_

