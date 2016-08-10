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
    this->defaults["crop"] = SettingV(0.00001f);
    
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
                            
    utils::Check(bottom.size() == BottomNodeNum(), "SoftmaxFuncLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "SoftmaxFuncLayer:top size problem.");
    axis = setting["axis"].iVal();
	crop = setting["crop"].fVal();
    utils::Check(0 < axis && axis < 4, "SoftmaxFuncLayer: axis error.");
      
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "SoftmaxFuncLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "SoftmaxFuncLayer:top size problem.");
    top[0]->Resize(bottom[0]->data.shape_, bottom[0]->length.shape_, true);
	if (show_info) {
		bottom[0]->PrintShape("bottom0");
		top[0]->PrintShape("top0");
	}
  }

  void checkNan(float *p, int l) {
    for (int i = 0; i < l; ++i) {
      assert(!std::isnan(p[i]));
    }
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    top[0]->length = F<op::identity>(bottom[0]->length); 
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Shape<4>       bottom_shape= bottom[0]->data.shape_;
    mshadow::Tensor<xpu, 4> top_data    = top[0]->data;
    
    int row = 1, col = 1;
    for (int i = 0; i < axis; ++i) {
      row *= int(bottom_shape[i]);
    }
    for (int i = axis; i < 4; ++i) {
      col *= int(bottom_shape[i]);
    }

    mshadow::Tensor<xpu, 2> input(bottom_data.dptr_, mshadow::Shape2(row, col));
    mshadow::Tensor<xpu, 2> output(top_data.dptr_,   mshadow::Shape2(row, col));
    mshadow::Softmax(output, input);
	int crop_count = 0;
    for (int i = 0; i < output.size(0); ++i) {
      for (int j = 0; j < output.size(1); ++j) {
        if (output[i][j] < crop) {
          output[i][j] = crop;
		  ++crop_count;
        }
      }
    }
#if DEBUG
    std::cout << "SoftmaxFuncLayer: WARNING, prob too small, crop. " << crop_count << std::endl;
    checkNan(top[0]->data.dptr_, top[0]->data.shape_.Size());
#endif
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    if(!this->prop_error[0])    return;
    using namespace mshadow::expr;
    mshadow::Shape<4>       bottom_shape= bottom[0]->data.shape_;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 4> top_data    = top[0]->data;
    mshadow::Tensor<xpu, 4> top_diff    = top[0]->diff;
    int row = 1, col = 1;
    for (int i = 0; i < axis; ++i) {
      row *= int(bottom_shape[i]);
    }
    for (int i = axis; i < 4; ++i) {
      col *= int(bottom_shape[i]);
    }
    mshadow::Tensor<xpu, 2> output_data(top_data.dptr_,   mshadow::Shape2(row, col));
    mshadow::Tensor<xpu, 2> input_diff(bottom_diff.dptr_, mshadow::Shape2(row, col));
    mshadow::Tensor<xpu, 2> output_diff(top_diff.dptr_,   mshadow::Shape2(row, col));

    for (int row_idx = 0; row_idx < row; ++row_idx) {
      float error_sum = 0.0f;
      for (int col_idx = 0; col_idx < col; ++col_idx) {
        error_sum += output_diff[row_idx][col_idx] * output_data[row_idx][col_idx];
      }
      for (int col_idx = 0; col_idx < col; ++col_idx) {
        input_diff[row_idx][col_idx] += (output_diff[row_idx][col_idx] - error_sum) * output_data[row_idx][col_idx];
      }
    }
#if DEBUG
    checkNan(bottom[0]->diff.dptr_, bottom[0]->diff.shape_.Size());
#endif
  }

  /*
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Shape<4>       bottom_shape= bottom[0]->data.shape_;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 4> top_data    = top[0]->data;
    mshadow::Tensor<xpu, 4> top_diff    = top[0]->diff;
    int row = 1, col = 1;
    for (int i = 0; i < axis; ++i) {
      row *= int(bottom_shape[i]);
    }
    for (int i = axis; i < 4; ++i) {
      col *= int(bottom_shape[i]);
    }
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
            input_diff[row_idx][col_idx] += top * (-(p_0*p_0) + p_0);
          } else {
            input_diff[row_idx][col_idx] += top * (-(p_0*p_1));
          }
        }
      }
    }
#if DEBUG
    checkNan(bottom[0]->diff.dptr_, bottom[0]->diff.shape_.Size());
#endif
  }
  */
  
 protected:
  int axis;
  float crop;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_SOFTMAX_FUNC_LAYER_INL_HPP_


