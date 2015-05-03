#ifndef TEXTNET_LAYER_SOFTMAX_FUNC_LAYER_INL_HPP_
#define TEXTNET_LAYER_SOFTMAX_FUNC_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu>
class SoftmaxFuncLayer : public Layer<xpu>{
 public:
  SoftmaxFuncLayer(LayerType type) { this->layer_type = type; }
  virtual ~SoftmaxFuncLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["axis"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
                            
    utils::Check(bottom.size() == BottomNodeNum(),
                 "SoftmaxFuncLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                 "SoftmaxFuncLayer:top size problem.");
    axis = setting["axis"].iVal();
    utils::Check(0 < axis && axis < 4, "SoftmaxFuncLayer: axis error.");
      
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                 "SoftmaxFuncLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                 "SoftmaxFuncLayer:top size problem.");
    // batch_size = bottom[0]->data.size(0);  
    top[0]->Resize(bottom[0]->data.shape_, true);
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom_shape= bottom[0]->data.shape_;
    mshadow::Tensor<xpu, 4> top_data    = top[0]->data;
    
    int row = 1, col = 1;
    for (int i = 0; i < axis; ++i) {
        row *= bottom_shape[i];
    }
    for (int i = axis; i < 4; ++i) {
        col *= bottom_shape[i];
    }

    mshadow::Tensor<xpu, 2> input(bottom_data.dptr_, mshadow::Shape2(row, col));
    mshadow::Tensor<xpu, 2> output(top_data.dptr_,   mshadow::Shape2(row, col));
    mshadow::Softmax(input, output);
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom_shape= bottom[0]->data.shape_;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 4> top_data    = top[0]->data;
    mshadow::Tensor<xpu, 4> top_diff    = top[0]->diff;
    int row = 1, col = 1;
    for (int i = 0; i < axis; ++i) {
        row *= bottom_shape[i];
    }
    for (int i = axis; i < 4; ++i) {
        col *= bottom_shape[i];
    }
    mshadow::Tensor<xpu, 2> input_data(bottom_data.dptr_, mshadow::Shape2(row, col));
    mshadow::Tensor<xpu, 2> output_data(top_data.dptr_,   mshadow::Shape2(row, col));
    mshadow::Tensor<xpu, 2> input_diff(bottom_diff.dptr_, mshadow::Shape2(row, col));
    mshadow::Tensor<xpu, 2> output_diff(top_diff.dptr_,   mshadow::Shape2(row, col));

    for (int row_idx = 0; row_idx < row; ++row_idx) {
      for (int col_idx = 0; col_idx < col; ++col_idx) {
        for (int jacobi_row_idx = 0; jacobi_row_idx < col; ++jacobi_row_idx) {
          float top = output_diff[row_idx][jacobi_row_idx];
          float p_0 = output_data[row_idx][col_idx];
          float p_1 = output_data[row_idx][jacobi_row_idx];
          if (jacobi_row_idx == col_idx) {
            input_diff[row_idx][col_idx] += top * (-p_0 * p_0 + p_0);
          } else {
            input_diff[row_idx][col_idx] += top * (-p_0 * p_1);
          }
        }
      }
    }
  }
  
 protected:
  int axis;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_SOFTMAX_FUNC_LAYER_INL_HPP_


