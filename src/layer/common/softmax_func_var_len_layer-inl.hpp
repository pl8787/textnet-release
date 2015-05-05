#ifndef TEXTNET_LAYER_SOFTMAX_FUNC_VAR_LEN_LAYER_INL_HPP_
#define TEXTNET_LAYER_SOFTMAX_FUNC_VAR_LEN_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

// this layer conduct softmax only on axis 2
template<typename xpu>
class SoftmaxFuncVarLenLayer : public Layer<xpu>{
 public:
  SoftmaxFuncVarLenLayer(LayerType type) { this->layer_type = type; }
  virtual ~SoftmaxFuncVarLenLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
                            
    utils::Check(bottom.size() == BottomNodeNum(),
                 "SoftmaxFuncVarLenLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                 "SoftmaxFuncVarLenLayer:top size problem.");
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                 "SoftmaxFuncVarLenLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                 "SoftmaxFuncVarLenLayer:top size problem.");
    top[0]->Resize(bottom[0]->data.shape_, true);
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Shape<4>       bottom_shape= bottom[0]->data.shape_;
    mshadow::Tensor<xpu, 3> bottom_data = bottom[0]->data_d3();
    mshadow::Tensor<xpu, 2> bottom_len  = bottom[0]->length;
    mshadow::Tensor<xpu, 3> top_data    = top[0]->data_d3();
    mshadow::Tensor<xpu, 2> top_len     = top[0]->length;
    top_len = F<op::identity_grad>(bottom_len);

    top_data = 0.f;
    for (int batch_idx = 0; batch_idx < bottom_shape[0]; ++batch_idx) {
      for (int seq_idx = 0; seq_idx < bottom_shape[1]; ++seq_idx) {
        int length = bottom_len[batch_idx][seq_idx] * bottom_shape[3];
        mshadow::Tensor<xpu,1> input  = bottom_data[batch_idx][seq_idx].Slice(0, length);
        mshadow::Tensor<xpu,1> output =    top_data[batch_idx][seq_idx].Slice(0, length);
        mshadow::Softmax(output, input);
        for (int i = 0; i < output.size(0); ++i) {
            if (output[i] < 0.00001f) {
              cout << "SoftmaxFuncVarLenLayer: WARNING, prob too small, crop." << endl;
              output[i] = 0.00001f;
            }
          }
        }
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Shape<4>       bottom_shape= bottom[0]->data.shape_;
    mshadow::Tensor<xpu, 3> bottom_diff = bottom[0]->diff_d3();
    mshadow::Tensor<xpu, 2> bottom_len  = bottom[0]->length;
    mshadow::Tensor<xpu, 3> top_data    = top[0]->data_d3();
    mshadow::Tensor<xpu, 3> top_diff    = top[0]->diff_d3();

    for (int batch_idx = 0; batch_idx < bottom_shape[0]; ++batch_idx) {
      for (int seq_idx = 0; seq_idx < bottom_shape[1]; ++seq_idx) {
        int length = bottom_len[batch_idx][seq_idx] * bottom_shape[3];
        mshadow::Tensor<xpu,1> output_data =    top_data[batch_idx][seq_idx].Slice(0, length);
        mshadow::Tensor<xpu,1> input_diff  = bottom_diff[batch_idx][seq_idx].Slice(0, length);
        mshadow::Tensor<xpu,1> output_diff =    top_diff[batch_idx][seq_idx].Slice(0, length);
        for (int col_idx = 0; col_idx < length; ++col_idx) {
          for (int jacobi_row_idx = 0; jacobi_row_idx < length; ++jacobi_row_idx) {
            float top = output_diff[jacobi_row_idx];
            float p_0 = output_data[col_idx];
            float p_1 = output_data[jacobi_row_idx];
            if (jacobi_row_idx == col_idx) {
              input_diff[col_idx] += top * (-p_0 * p_0 + p_0);
            } else {
              input_diff[col_idx] += top * (-p_0 * p_1);
            }
          }
        }
      }
    }
  }
  
 // protected:
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_SOFTMAX_VAR_LEN_FUNC_LAYER_INL_HPP_


